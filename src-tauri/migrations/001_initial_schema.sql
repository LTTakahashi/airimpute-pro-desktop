-- Initial database schema with full ACID compliance
-- Supports comprehensive data persistence for air quality imputation

-- NOTE: PRAGMA statements are set in the Database::new() function
-- to avoid SQLite transaction errors during migration

-- Projects table - stores project metadata
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_accessed TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    version INTEGER NOT NULL DEFAULT 1,
    metadata JSON
);

-- Datasets table - stores dataset information
CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY NOT NULL,
    project_id TEXT NOT NULL,
    name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_hash TEXT NOT NULL,  -- For data integrity verification
    rows INTEGER NOT NULL,
    columns INTEGER NOT NULL,
    missing_count INTEGER NOT NULL DEFAULT 0,
    missing_percentage REAL NOT NULL DEFAULT 0.0,
    statistics JSON NOT NULL,  -- Comprehensive statistics
    column_metadata JSON NOT NULL,  -- Column names, types, etc.
    temporal_info JSON,  -- Time series specific info
    spatial_info JSON,  -- Spatial data specific info
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);

-- Imputation jobs table - tracks all imputation operations
CREATE TABLE IF NOT EXISTS imputation_jobs (
    id TEXT PRIMARY KEY NOT NULL,
    project_id TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    method TEXT NOT NULL,
    parameters JSON NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    priority INTEGER NOT NULL DEFAULT 5,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    duration_ms INTEGER,
    error_message TEXT,
    error_details JSON,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
    FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
);

-- Imputation results table - stores imputation outputs
CREATE TABLE IF NOT EXISTS imputation_results (
    id TEXT PRIMARY KEY NOT NULL,
    job_id TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    imputed_data_path TEXT NOT NULL,  -- Path to imputed data file
    imputed_data_hash TEXT NOT NULL,  -- For integrity verification
    quality_metrics JSON NOT NULL,  -- RMSE, MAE, etc.
    uncertainty_metrics JSON,  -- Confidence intervals, variance
    validation_results JSON,  -- Cross-validation results
    method_specific_outputs JSON,  -- Method-specific results
    data_preview JSON,  -- Small preview of imputed data
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (job_id) REFERENCES imputation_jobs(id) ON DELETE CASCADE,
    FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
);

-- Method performance table - tracks method performance over time
CREATE TABLE IF NOT EXISTS method_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    method TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    rmse REAL,
    mae REAL,
    r2_score REAL,
    execution_time_ms INTEGER NOT NULL,
    memory_usage_mb REAL,
    cpu_usage_percent REAL,
    gpu_usage_percent REAL,
    parameters JSON NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE,
    FOREIGN KEY (job_id) REFERENCES imputation_jobs(id) ON DELETE CASCADE
);

-- User preferences table - stores user settings
CREATE TABLE IF NOT EXISTS user_preferences (
    key TEXT PRIMARY KEY NOT NULL,
    value JSON NOT NULL,
    category TEXT NOT NULL,
    description TEXT,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Audit log table - tracks all database operations for compliance
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    table_name TEXT NOT NULL,
    operation TEXT NOT NULL CHECK (operation IN ('INSERT', 'UPDATE', 'DELETE')),
    record_id TEXT NOT NULL,
    old_values JSON,
    new_values JSON,
    user_id TEXT,
    ip_address TEXT,
    user_agent TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Cache table - for performance optimization
CREATE TABLE IF NOT EXISTS cache (
    key TEXT PRIMARY KEY NOT NULL,
    value BLOB NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    access_count INTEGER NOT NULL DEFAULT 0,
    last_accessed TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_datasets_project_id ON datasets(project_id);
CREATE INDEX idx_datasets_created_at ON datasets(created_at);
CREATE INDEX idx_imputation_jobs_project_id ON imputation_jobs(project_id);
CREATE INDEX idx_imputation_jobs_dataset_id ON imputation_jobs(dataset_id);
CREATE INDEX idx_imputation_jobs_status ON imputation_jobs(status);
CREATE INDEX idx_imputation_jobs_created_at ON imputation_jobs(created_at);
CREATE INDEX idx_imputation_results_job_id ON imputation_results(job_id);
CREATE INDEX idx_method_performance_method ON method_performance(method);
CREATE INDEX idx_method_performance_dataset_id ON method_performance(dataset_id);
CREATE INDEX idx_audit_log_table_name ON audit_log(table_name);
CREATE INDEX idx_audit_log_created_at ON audit_log(created_at);
CREATE INDEX idx_cache_expires_at ON cache(expires_at);

-- Triggers for automatic timestamp updates
CREATE TRIGGER update_projects_timestamp 
    AFTER UPDATE ON projects
    FOR EACH ROW
BEGIN
    UPDATE projects SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER update_datasets_timestamp 
    AFTER UPDATE ON datasets
    FOR EACH ROW
BEGIN
    UPDATE datasets SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER update_imputation_jobs_timestamp 
    AFTER UPDATE ON imputation_jobs
    FOR EACH ROW
BEGIN
    UPDATE imputation_jobs SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- Triggers for audit logging
CREATE TRIGGER audit_projects_insert
    AFTER INSERT ON projects
    FOR EACH ROW
BEGIN
    INSERT INTO audit_log (table_name, operation, record_id, new_values)
    VALUES ('projects', 'INSERT', NEW.id, json_object(
        'id', NEW.id,
        'name', NEW.name,
        'description', NEW.description
    ));
END;

CREATE TRIGGER audit_projects_update
    AFTER UPDATE ON projects
    FOR EACH ROW
BEGIN
    INSERT INTO audit_log (table_name, operation, record_id, old_values, new_values)
    VALUES ('projects', 'UPDATE', NEW.id, 
        json_object('name', OLD.name, 'description', OLD.description),
        json_object('name', NEW.name, 'description', NEW.description)
    );
END;

CREATE TRIGGER audit_projects_delete
    AFTER DELETE ON projects
    FOR EACH ROW
BEGIN
    INSERT INTO audit_log (table_name, operation, record_id, old_values)
    VALUES ('projects', 'DELETE', OLD.id, json_object(
        'id', OLD.id,
        'name', OLD.name,
        'description', OLD.description
    ));
END;

-- Views for common queries
CREATE VIEW v_active_jobs AS
    SELECT 
        j.*,
        d.name as dataset_name,
        p.name as project_name
    FROM imputation_jobs j
    JOIN datasets d ON j.dataset_id = d.id
    JOIN projects p ON j.project_id = p.id
    WHERE j.status IN ('pending', 'running')
    ORDER BY j.priority DESC, j.created_at ASC;

CREATE VIEW v_method_performance_summary AS
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
    GROUP BY method;

-- Initial system preferences
INSERT INTO user_preferences (key, value, category, description) VALUES
    ('auto_save_interval', '"300"', 'system', 'Auto-save interval in seconds'),
    ('max_memory_usage', '"4096"', 'system', 'Maximum memory usage in MB'),
    ('parallel_jobs', '"4"', 'system', 'Maximum parallel imputation jobs'),
    ('cache_ttl', '"3600"', 'system', 'Cache time-to-live in seconds'),
    ('theme', '"light"', 'ui', 'User interface theme'),
    ('language', '"en"', 'ui', 'User interface language');