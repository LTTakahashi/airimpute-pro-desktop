use std::path::Path;
use anyhow::{Result, Context};
use sqlx::{SqlitePool, sqlite::SqlitePoolOptions, Transaction, Sqlite};
use uuid::Uuid;
use chrono::Utc;
use serde_json::Value as JsonValue;
use sha2::{Sha256, Digest};
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::core::imputation_result::ImputationResult as CoreImputationResult;
use crate::core::data::DataStatistics;

pub mod models;
pub mod repository;

use models::*;

/// Database connection and operations with full ACID compliance
pub struct Database {
    pub(crate) pool: SqlitePool,
    cache: Arc<RwLock<std::collections::HashMap<String, Vec<u8>>>>,
}

/// Transaction wrapper for ACID operations
pub struct DbTransaction<'a> {
    pub(crate) tx: Transaction<'a, Sqlite>,
}

impl<'a> DbTransaction<'a> {
    /// Commit the transaction
    pub async fn commit(self) -> Result<()> {
        self.tx.commit().await.context("Failed to commit transaction")?;
        Ok(())
    }
    
    /// Rollback the transaction
    pub async fn rollback(self) -> Result<()> {
        self.tx.rollback().await.context("Failed to rollback transaction")?;
        Ok(())
    }
}

impl Database {
    /// Create new database connection with ACID compliance
    pub async fn new(path: &Path) -> Result<Self> {
        Self::new_with_migrations(path, Path::new("./migrations")).await
    }
    
    /// Create new database connection with custom migrations path
    pub async fn new_with_migrations(path: &Path, migrations_path: &Path) -> Result<Self> {
        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        
        let connection_string = format!("sqlite://{}?mode=rwc", path.display());
        
        let pool = SqlitePoolOptions::new()
            .max_connections(10)
            .min_connections(2)
            .acquire_timeout(std::time::Duration::from_secs(10))
            .idle_timeout(std::time::Duration::from_secs(600))
            .max_lifetime(std::time::Duration::from_secs(1800))
            .connect(&connection_string)
            .await?;
        
        // Configure SQLite for optimal ACID compliance
        sqlx::query("PRAGMA foreign_keys = ON")
            .execute(&pool)
            .await?;
        sqlx::query("PRAGMA journal_mode = WAL")
            .execute(&pool)
            .await?;
        sqlx::query("PRAGMA synchronous = FULL")
            .execute(&pool)
            .await?;
        sqlx::query("PRAGMA temp_store = MEMORY")
            .execute(&pool)
            .await?;
        
        // Run migrations using runtime loading instead of compile-time macro
        let migrator = sqlx::migrate::Migrator::new(migrations_path)
            .await
            .context("Failed to create migrator")?;
        migrator.run(&pool)
            .await
            .context("Failed to run migrations")?;
        
        Ok(Self { 
            pool,
            cache: Arc::new(RwLock::new(std::collections::HashMap::new())),
        })
    }
    
    /// Begin a new transaction
    pub async fn begin_transaction(&self) -> Result<DbTransaction<'_>> {
        let tx = self.pool.begin().await?;
        Ok(DbTransaction { tx })
    }
    
    /// Save dataset metadata with transaction support
    pub async fn save_dataset_metadata(
        &self,
        project_id: &Uuid,
        name: &str,
        path: &Path,
        statistics: &DataStatistics,
        column_metadata: JsonValue,
    ) -> Result<Dataset> {
        let id = Uuid::new_v4().to_string();
        let file_hash = self.calculate_file_hash(path).await?;
        
        let dataset = sqlx::query_as::<_, Dataset>(
            r#"
            INSERT INTO datasets (
                id, project_id, name, file_path, file_hash,
                rows, columns, missing_count, missing_percentage,
                statistics, column_metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING *
            "#
        )
        .bind(&id)
        .bind(project_id.to_string())
        .bind(name)
        .bind(path.to_string_lossy().to_string())
        .bind(&file_hash)
        .bind(statistics.total_rows as i32)
        .bind(statistics.total_columns as i32)
        .bind(statistics.total_missing as i32)
        .bind(statistics.missing_percentage)
        .bind(serde_json::to_value(statistics)?)
        .bind(column_metadata)
        .fetch_one(&self.pool)
        .await?;
        
        Ok(dataset)
    }
    
    /// Calculate file hash for integrity verification
    pub(crate) async fn calculate_file_hash(&self, path: &Path) -> Result<String> {
        let content = tokio::fs::read(path).await?;
        let mut hasher = Sha256::new();
        hasher.update(&content);
        Ok(format!("{:x}", hasher.finalize()))
    }
    
    /// Save imputation result with full metrics
    pub async fn save_imputation_result(
        &self,
        job_id: &Uuid,
        result: &CoreImputationResult,
        imputed_data_path: &Path,
    ) -> Result<ImputationResult> {
        let id = Uuid::new_v4().to_string();
        let file_hash = self.calculate_file_hash(imputed_data_path).await?;
        
        // Get job details to find dataset_id
        let job = self.get_imputation_job(job_id).await?;
        
        // Extract quality metrics from all columns
        let quality_metrics = serde_json::to_value(&result.metrics)?;
        
        let uncertainty_metrics = if let Some(uncertainty) = &result.uncertainty {
            Some(serde_json::to_value(uncertainty)?)
        } else {
            None
        };
        
        let imputation_result = sqlx::query_as::<_, ImputationResult>(
            r#"
            INSERT INTO imputation_results (
                id, job_id, dataset_id, imputed_data_path,
                imputed_data_hash, quality_metrics, uncertainty_metrics,
                validation_results, method_specific_outputs
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING *
            "#
        )
        .bind(&id)
        .bind(job_id.to_string())
        .bind(&job.dataset_id)
        .bind(imputed_data_path.to_string_lossy().to_string())
        .bind(&file_hash)
        .bind(quality_metrics)
        .bind(uncertainty_metrics)
        .bind(result.validation_results.as_ref().and_then(|v| serde_json::to_value(v).ok()))
        .bind(result.method_outputs.as_ref().and_then(|v| serde_json::to_value(v).ok()))
        .fetch_one(&self.pool)
        .await?;
        
        // Update job status
        self.update_job_status(job_id, JobStatus::Completed).await?;
        
        // Save performance metrics
        self.save_method_performance(&job, result).await?;
        
        Ok(imputation_result)
    }
    
    /// Create a new project
    pub async fn create_project(&self, name: &str, description: Option<&str>) -> Result<Project> {
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
        .fetch_one(&self.pool)
        .await?;
        
        Ok(project)
    }
    
    /// Get project by ID
    pub async fn get_project(&self, id: &Uuid) -> Result<Project> {
        let project = sqlx::query_as::<_, Project>(
            "SELECT * FROM projects WHERE id = ?"
        )
        .bind(id.to_string())
        .fetch_one(&self.pool)
        .await?;
        
        // Update last accessed time
        sqlx::query("UPDATE projects SET last_accessed = CURRENT_TIMESTAMP WHERE id = ?")
            .bind(id.to_string())
            .execute(&self.pool)
            .await?;
        
        Ok(project)
    }
    
    /// List all projects
    pub async fn list_projects(&self) -> Result<Vec<Project>> {
        let projects = sqlx::query_as::<_, Project>(
            "SELECT * FROM projects ORDER BY last_accessed DESC"
        )
        .fetch_all(&self.pool)
        .await?;
        
        Ok(projects)
    }
    
    /// Create a new imputation job
    pub async fn create_imputation_job(
        &self,
        project_id: &Uuid,
        dataset_id: &Uuid,
        method: &str,
        parameters: JsonValue,
        priority: Option<i32>,
    ) -> Result<ImputationJob> {
        let id = Uuid::new_v4().to_string();
        
        let job = sqlx::query_as::<_, ImputationJob>(
            r#"
            INSERT INTO imputation_jobs (
                id, project_id, dataset_id, method,
                parameters, status, priority
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            RETURNING *
            "#
        )
        .bind(&id)
        .bind(project_id.to_string())
        .bind(dataset_id.to_string())
        .bind(method)
        .bind(parameters)
        .bind(JobStatus::Pending.to_string())
        .bind(priority.unwrap_or(5))
        .fetch_one(&self.pool)
        .await?;
        
        Ok(job)
    }
    
    /// Get imputation job by ID
    pub async fn get_imputation_job(&self, id: &Uuid) -> Result<ImputationJob> {
        let job = sqlx::query_as::<_, ImputationJob>(
            "SELECT * FROM imputation_jobs WHERE id = ?"
        )
        .bind(id.to_string())
        .fetch_one(&self.pool)
        .await?;
        
        Ok(job)
    }
    
    /// Update job status
    pub async fn update_job_status(&self, id: &Uuid, status: JobStatus) -> Result<()> {
        let query = match status {
            JobStatus::Running => {
                sqlx::query(
                    "UPDATE imputation_jobs SET status = ?, started_at = CURRENT_TIMESTAMP WHERE id = ?"
                )
                .bind(status.to_string())
                .bind(id.to_string())
            }
            JobStatus::Completed => {
                sqlx::query(
                    r#"
                    UPDATE imputation_jobs 
                    SET status = ?, 
                        completed_at = CURRENT_TIMESTAMP,
                        duration_ms = CAST((julianday(CURRENT_TIMESTAMP) - julianday(started_at)) * 86400000 AS INTEGER)
                    WHERE id = ?
                    "#
                )
                .bind(status.to_string())
                .bind(id.to_string())
            }
            _ => {
                sqlx::query(
                    "UPDATE imputation_jobs SET status = ? WHERE id = ?"
                )
                .bind(status.to_string())
                .bind(id.to_string())
            }
        };
        
        query.execute(&self.pool).await?;
        Ok(())
    }
    
    /// Save method performance metrics
    async fn save_method_performance(
        &self,
        job: &ImputationJob,
        result: &CoreImputationResult,
    ) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO method_performance (
                method, dataset_id, job_id, rmse, mae, r2_score,
                execution_time_ms, memory_usage_mb, parameters
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#
        )
        .bind(&job.method)
        .bind(&job.dataset_id)
        .bind(&job.id)
        // Calculate average metrics across all columns
        .bind(if result.metrics.is_empty() { None } else {
            Some(result.metrics.values().map(|m| m.rmse).sum::<f64>() / result.metrics.len() as f64)
        })
        .bind(if result.metrics.is_empty() { None } else {
            Some(result.metrics.values().map(|m| m.mae).sum::<f64>() / result.metrics.len() as f64)
        })
        .bind(if result.metrics.is_empty() { None } else {
            result.metrics.values().filter_map(|m| m.r2).next()
        })
        .bind(job.duration_ms.unwrap_or(0))
        .bind(result.memory_usage_mb)
        .bind(&job.parameters)
        .execute(&self.pool)
        .await?;
        
        Ok(())
    }
    
    /// Get active jobs
    pub async fn get_active_jobs(&self) -> Result<Vec<ActiveJobView>> {
        let jobs = sqlx::query_as::<_, ActiveJobView>(
            "SELECT * FROM v_active_jobs"
        )
        .fetch_all(&self.pool)
        .await?;
        
        Ok(jobs)
    }
    
    /// Get method performance summary
    pub async fn get_method_performance_summary(&self) -> Result<Vec<MethodPerformanceSummary>> {
        let summary = sqlx::query_as::<_, MethodPerformanceSummary>(
            "SELECT * FROM v_method_performance_summary"
        )
        .fetch_all(&self.pool)
        .await?;
        
        Ok(summary)
    }
    
    /// Get user preference
    pub async fn get_preference(&self, key: &str) -> Result<Option<JsonValue>> {
        let pref = sqlx::query_as::<_, UserPreference>(
            "SELECT * FROM user_preferences WHERE key = ?"
        )
        .bind(key)
        .fetch_optional(&self.pool)
        .await?;
        
        Ok(pref.map(|p| p.value))
    }
    
    /// Set user preference
    pub async fn set_preference(&self, key: &str, value: JsonValue, category: &str) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO user_preferences (key, value, category)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = CURRENT_TIMESTAMP
            "#
        )
        .bind(key)
        .bind(value)
        .bind(category)
        .execute(&self.pool)
        .await?;
        
        Ok(())
    }
    
    /// Cache operations
    pub async fn cache_get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        // Check in-memory cache first
        {
            let cache = self.cache.read().await;
            if let Some(value) = cache.get(key) {
                return Ok(Some(value.clone()));
            }
        }
        
        // Check database cache
        let entry = sqlx::query_as::<_, CacheEntry>(
            r#"
            SELECT * FROM cache 
            WHERE key = ? AND expires_at > CURRENT_TIMESTAMP
            "#
        )
        .bind(key)
        .fetch_optional(&self.pool)
        .await?;
        
        if let Some(entry) = entry {
            // Update access count and last accessed
            sqlx::query(
                r#"
                UPDATE cache 
                SET access_count = access_count + 1,
                    last_accessed = CURRENT_TIMESTAMP
                WHERE key = ?
                "#
            )
            .bind(key)
            .execute(&self.pool)
            .await?;
            
            // Store in memory cache
            {
                let mut cache = self.cache.write().await;
                cache.insert(key.to_string(), entry.value.clone());
            }
            
            Ok(Some(entry.value))
        } else {
            Ok(None)
        }
    }
    
    pub async fn cache_set(&self, key: &str, value: Vec<u8>, ttl_seconds: i64) -> Result<()> {
        let expires_at = Utc::now() + chrono::Duration::seconds(ttl_seconds);
        
        sqlx::query(
            r#"
            INSERT INTO cache (key, value, expires_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                expires_at = excluded.expires_at,
                access_count = 0,
                last_accessed = CURRENT_TIMESTAMP
            "#
        )
        .bind(key)
        .bind(&value)
        .bind(expires_at)
        .execute(&self.pool)
        .await?;
        
        // Update memory cache
        {
            let mut cache = self.cache.write().await;
            cache.insert(key.to_string(), value);
        }
        
        Ok(())
    }
    
    /// Clean expired cache entries
    pub async fn clean_expired_cache(&self) -> Result<()> {
        sqlx::query(
            "DELETE FROM cache WHERE expires_at < CURRENT_TIMESTAMP"
        )
        .execute(&self.pool)
        .await?;
        
        Ok(())
    }
    
    /// Perform database optimization
    pub async fn optimize(&self) -> Result<()> {
        sqlx::query("VACUUM").execute(&self.pool).await?;
        sqlx::query("ANALYZE").execute(&self.pool).await?;
        Ok(())
    }
    
    /// Close database connection
    pub async fn close(&self) -> Result<()> {
        self.pool.close().await;
        Ok(())
    }
}