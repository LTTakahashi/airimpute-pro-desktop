use anyhow::Result;
use uuid::Uuid;
use async_trait::async_trait;
use std::sync::Arc;

use super::{Database, models::*};
use crate::core::data::DataStatistics;

/// Repository trait for data access abstraction
#[async_trait]
pub trait Repository<T> {
    async fn create(&self, entity: T) -> Result<T>;
    async fn get(&self, id: &Uuid) -> Result<T>;
    async fn update(&self, entity: T) -> Result<T>;
    async fn delete(&self, id: &Uuid) -> Result<()>;
    async fn list(&self) -> Result<Vec<T>>;
}

/// Project repository
pub struct ProjectRepository {
    db: Arc<Database>,
}

impl ProjectRepository {
    pub fn new(db: Arc<Database>) -> Self {
        Self { db }
    }
    
    pub async fn create_with_transaction(
        &self,
        name: &str,
        description: Option<&str>,
    ) -> Result<Project> {
        let mut tx = self.db.pool.begin().await?;
        
        let id = Uuid::new_v4().to_string();
        let project = sqlx::query_as::<_, Project>(
            r#"
            INSERT INTO projects (id, name, description)
            VALUES (?, ?, ?)
            RETURNING *
            "#
        )
        .bind(&id)
        .bind(name)
        .bind(description)
        .fetch_one(&mut *tx)
        .await?;
        
        tx.commit().await?;
        Ok(project)
    }
    
    pub async fn update_metadata(
        &self,
        id: &Uuid,
        metadata: serde_json::Value,
    ) -> Result<()> {
        sqlx::query(
            r#"
            UPDATE projects 
            SET metadata = ?, version = version + 1
            WHERE id = ?
            "#
        )
        .bind(metadata)
        .bind(id.to_string())
        .execute(&self.db.pool)
        .await?;
        
        Ok(())
    }
}

/// Dataset repository with complex operations
pub struct DatasetRepository {
    db: Arc<Database>,
}

impl DatasetRepository {
    pub fn new(db: Arc<Database>) -> Self {
        Self { db }
    }
    
    pub async fn create_with_validation(
        &self,
        project_id: &Uuid,
        name: &str,
        path: &std::path::Path,
        statistics: &DataStatistics,
        column_metadata: serde_json::Value,
    ) -> Result<Dataset> {
        // Validate that project exists
        let _ = sqlx::query("SELECT id FROM projects WHERE id = ?")
            .bind(project_id.to_string())
            .fetch_one(&self.db.pool)
            .await?;
        
        // Check for duplicate datasets
        #[derive(sqlx::FromRow)]
        struct CountResult {
            count: i64,
        }
        
        let existing = sqlx::query_as::<_, CountResult>(
            "SELECT COUNT(*) as count FROM datasets WHERE project_id = ? AND name = ?"
        )
        .bind(project_id.to_string())
        .bind(name)
        .fetch_one(&self.db.pool)
        .await?;
        
        if existing.count > 0 {
            return Err(anyhow::anyhow!("Dataset with name '{}' already exists", name));
        }
        
        // Create dataset
        self.db.save_dataset_metadata(
            project_id,
            name,
            path,
            statistics,
            column_metadata,
        ).await
    }
    
    pub async fn get_by_project(&self, project_id: &Uuid) -> Result<Vec<Dataset>> {
        let datasets = sqlx::query_as::<_, Dataset>(
            "SELECT * FROM datasets WHERE project_id = ? ORDER BY created_at DESC"
        )
        .bind(project_id.to_string())
        .fetch_all(&self.db.pool)
        .await?;
        
        Ok(datasets)
    }
    
    pub async fn verify_integrity(&self, id: &Uuid) -> Result<bool> {
        let dataset = sqlx::query_as::<_, Dataset>(
            "SELECT * FROM datasets WHERE id = ?"
        )
        .bind(id.to_string())
        .fetch_one(&self.db.pool)
        .await?;
        
        // Calculate current file hash
        let current_hash = self.db.calculate_file_hash(
            std::path::Path::new(&dataset.file_path)
        ).await?;
        
        Ok(current_hash == dataset.file_hash)
    }
}

/// Imputation job repository with queue management
pub struct JobRepository {
    db: Arc<Database>,
}

impl JobRepository {
    pub fn new(db: Arc<Database>) -> Self {
        Self { db }
    }
    
    pub async fn get_next_pending(&self) -> Result<Option<ImputationJob>> {
        // Use transaction to ensure atomic operation
        let mut tx = self.db.pool.begin().await?;
        
        // Get highest priority pending job
        let job = sqlx::query_as::<_, ImputationJob>(
            r#"
            SELECT * FROM imputation_jobs
            WHERE status = 'pending'
            ORDER BY priority DESC, created_at ASC
            LIMIT 1
            FOR UPDATE
            "#
        )
        .fetch_optional(&mut *tx)
        .await?;
        
        if let Some(job) = job {
            // Mark as running
            sqlx::query(
                r#"
                UPDATE imputation_jobs 
                SET status = 'running', started_at = CURRENT_TIMESTAMP
                WHERE id = ?
                "#
            )
            .bind(&job.id)
            .execute(&mut *tx)
            .await?;
            
            tx.commit().await?;
            Ok(Some(job))
        } else {
            tx.rollback().await?;
            Ok(None)
        }
    }
    
    pub async fn update_job_status(
        &self,
        id: &Uuid,
        status: JobStatus,
    ) -> Result<()> {
        sqlx::query(
            r#"
            UPDATE imputation_jobs 
            SET status = ?
            WHERE id = ?
            "#
        )
        .bind(status.to_string())
        .bind(id.to_string())
        .execute(&self.db.pool)
        .await?;
        
        Ok(())
    }
    
    pub async fn fail_job(
        &self,
        id: &Uuid,
        error_message: &str,
        error_details: Option<serde_json::Value>,
    ) -> Result<()> {
        sqlx::query(
            r#"
            UPDATE imputation_jobs 
            SET status = 'failed',
                completed_at = CURRENT_TIMESTAMP,
                error_message = ?,
                error_details = ?,
                duration_ms = CAST((julianday(CURRENT_TIMESTAMP) - julianday(started_at)) * 86400000 AS INTEGER)
            WHERE id = ?
            "#
        )
        .bind(error_message)
        .bind(error_details)
        .bind(id.to_string())
        .execute(&self.db.pool)
        .await?;
        
        Ok(())
    }
    
    pub async fn get_job_history(
        &self,
        dataset_id: &Uuid,
        limit: i32,
    ) -> Result<Vec<ImputationJob>> {
        let jobs = sqlx::query_as::<_, ImputationJob>(
            r#"
            SELECT * FROM imputation_jobs
            WHERE dataset_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            "#
        )
        .bind(dataset_id.to_string())
        .bind(limit)
        .fetch_all(&self.db.pool)
        .await?;
        
        Ok(jobs)
    }
}

/// Performance analytics repository
pub struct AnalyticsRepository {
    db: Arc<Database>,
}

impl AnalyticsRepository {
    pub fn new(db: Arc<Database>) -> Self {
        Self { db }
    }
    
    pub async fn get_method_comparison(
        &self,
        dataset_id: &Uuid,
    ) -> Result<Vec<MethodPerformanceSummary>> {
        let summary = sqlx::query_as::<_, MethodPerformanceSummary>(
            r#"
            SELECT 
                method,
                COUNT(*) as run_count,
                AVG(rmse) as avg_rmse,
                MIN(rmse) as min_rmse,
                MAX(rmse) as max_rmse,
                AVG(mae) as avg_mae,
                AVG(r2_score) as avg_r2,
                AVG(execution_time_ms) as avg_time_ms,
                AVG(memory_usage_mb) as avg_memory_mb
            FROM method_performance
            WHERE dataset_id = ?
            GROUP BY method
            ORDER BY avg_rmse ASC
            "#
        )
        .bind(dataset_id.to_string())
        .fetch_all(&self.db.pool)
        .await?;
        
        Ok(summary)
    }
    
    pub async fn get_performance_trends(
        &self,
        method: &str,
        days: i32,
    ) -> Result<Vec<MethodPerformance>> {
        let trends = sqlx::query_as::<_, MethodPerformance>(
            r#"
            SELECT * FROM method_performance
            WHERE method = ?
              AND created_at >= datetime('now', ? || ' days')
            ORDER BY created_at ASC
            "#
        )
        .bind(method)
        .bind(format!("-{}", days))
        .fetch_all(&self.db.pool)
        .await?;
        
        Ok(trends)
    }
}

/// Unit of Work pattern for complex transactions
pub struct UnitOfWork {
    db: Arc<Database>,
}

impl UnitOfWork {
    pub fn new(db: Arc<Database>) -> Self {
        Self { db }
    }
    
    pub async fn execute_imputation_workflow<F, Fut>(
        &self,
        job_id: &Uuid,
        operation: F,
    ) -> Result<()>
    where
        F: FnOnce(Arc<Database>) -> Fut,
        Fut: std::future::Future<Output = Result<()>>,
    {
        // Create job repository for job operations
        let job_repo = JobRepository::new(self.db.clone());
        
        // Start transaction
        let mut tx = self.db.pool.begin().await?;
        
        // Update job status to running
        job_repo.update_job_status(job_id, JobStatus::Running).await?;
        
        // Execute the operation
        match operation(self.db.clone()).await {
            Ok(_) => {
                // Commit on success
                tx.commit().await?;
                Ok(())
            }
            Err(e) => {
                // Rollback on failure
                tx.rollback().await?;
                
                // Update job status to failed
                let _ = job_repo.fail_job(
                    job_id,
                    &e.to_string(),
                    Some(serde_json::json!({
                        "error_type": "workflow_failure",
                        "details": e.to_string()
                    }))
                ).await;
                
                Err(e)
            }
        }
    }
}