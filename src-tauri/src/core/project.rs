use uuid::Uuid;
use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};
use std::path::PathBuf;
use anyhow::Result;

use crate::db::Database;

/// Project structure for organizing work
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Project {
    pub id: Uuid,
    pub name: String,
    pub description: String,
    pub created_at: DateTime<Utc>,
    pub modified_at: DateTime<Utc>,
    pub datasets: Vec<Uuid>,
    pub imputation_jobs: Vec<Uuid>,
    pub exports: Vec<ExportRecord>,
    pub notes: String,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportRecord {
    pub id: Uuid,
    pub dataset_id: Uuid,
    pub job_id: Option<Uuid>,
    pub path: PathBuf,
    pub format: String,
    pub exported_at: DateTime<Utc>,
}

impl Project {
    pub fn new(name: String, description: String) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            name,
            description,
            created_at: now,
            modified_at: now,
            datasets: Vec::new(),
            imputation_jobs: Vec::new(),
            exports: Vec::new(),
            notes: String::new(),
            tags: Vec::new(),
        }
    }
    
    pub async fn save(&self, db: &Database) -> Result<()> {
        // Save project to database
        Ok(())
    }
    
    pub async fn load(db: &Database, id: Uuid) -> Result<Self> {
        // Load project from database
        todo!()
    }
}