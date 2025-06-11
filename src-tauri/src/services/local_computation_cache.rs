// Local computation cache - stores results on disk for offline use
use std::path::{Path, PathBuf};
use std::fs::{self, File};
use std::io::{Read, Write};
use serde::{Serialize, Deserialize};
use sha2::{Sha256, Digest};
use chrono::{DateTime, Utc, Duration};
use std::collections::HashMap;
use parking_lot::RwLock;
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    pub key: String,
    pub operation: String,
    pub created_at: DateTime<Utc>,
    pub last_accessed: DateTime<Utc>,
    pub access_count: u32,
    pub size_bytes: u64,
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub total_entries: usize,
    pub total_size_mb: f64,
    pub hit_rate: f32,
    pub most_used: Vec<(String, u32)>,
    pub space_saved_mb: f64,
}

pub struct LocalComputationCache {
    cache_dir: PathBuf,
    index: Arc<RwLock<HashMap<String, CacheEntry>>>,
    max_size_mb: f64,
    max_age_days: i64,
    hits: Arc<RwLock<u64>>,
    misses: Arc<RwLock<u64>>,
}

impl LocalComputationCache {
    pub fn new(app_data_dir: &Path, max_size_mb: f64) -> anyhow::Result<Self> {
        let cache_dir = app_data_dir.join("computation_cache");
        fs::create_dir_all(&cache_dir)?;
        
        let index_path = cache_dir.join("cache_index.json");
        let index = if index_path.exists() {
            let content = fs::read_to_string(&index_path)?;
            serde_json::from_str(&content).unwrap_or_default()
        } else {
            HashMap::new()
        };
        
        let cache = Self {
            cache_dir,
            index: Arc::new(RwLock::new(index)),
            max_size_mb,
            max_age_days: 30,
            hits: Arc::new(RwLock::new(0)),
            misses: Arc::new(RwLock::new(0)),
        };
        
        // Clean up old entries on startup
        cache.cleanup_old_entries()?;
        
        Ok(cache)
    }
    
    /// Generate cache key from operation parameters
    pub fn generate_key(
        operation: &str,
        data_hash: &str,
        method: &str,
        parameters: &serde_json::Value,
    ) -> String {
        let mut hasher = Sha256::new();
        hasher.update(operation.as_bytes());
        hasher.update(data_hash.as_bytes());
        hasher.update(method.as_bytes());
        hasher.update(parameters.to_string().as_bytes());
        
        format!("{:x}", hasher.finalize())
    }
    
    /// Store computation result in cache
    pub async fn store<T: Serialize>(
        &self,
        key: String,
        operation: String,
        result: &T,
        metadata: HashMap<String, serde_json::Value>,
    ) -> anyhow::Result<()> {
        // Serialize result
        let data = bincode::serialize(result)?;
        let size_bytes = data.len() as u64;
        
        // Check if we need to make space
        self.ensure_space_available(size_bytes).await?;
        
        // Write to disk
        let file_path = self.cache_dir.join(&key);
        let mut file = File::create(&file_path)?;
        file.write_all(&data)?;
        
        // Update index
        let entry = CacheEntry {
            key: key.clone(),
            operation,
            created_at: Utc::now(),
            last_accessed: Utc::now(),
            access_count: 0,
            size_bytes,
            metadata,
        };
        
        self.index.write().insert(key, entry);
        self.save_index()?;
        
        Ok(())
    }
    
    /// Retrieve computation result from cache
    pub async fn retrieve<T: for<'de> Deserialize<'de>>(
        &self,
        key: &str,
    ) -> anyhow::Result<Option<T>> {
        let mut index = self.index.write();
        
        if let Some(entry) = index.get_mut(key) {
            // Update access stats
            entry.last_accessed = Utc::now();
            entry.access_count += 1;
            
            let file_path = self.cache_dir.join(key);
            if file_path.exists() {
                let mut file = File::open(&file_path)?;
                let mut data = Vec::new();
                file.read_to_end(&mut data)?;
                
                let result: T = bincode::deserialize(&data)?;
                
                *self.hits.write() += 1;
                drop(index); // Release lock before saving
                self.save_index()?;
                
                Ok(Some(result))
            } else {
                // File missing, remove from index
                index.remove(key);
                *self.misses.write() += 1;
                Ok(None)
            }
        } else {
            *self.misses.write() += 1;
            Ok(None)
        }
    }
    
    /// Check if result exists in cache
    pub fn exists(&self, key: &str) -> bool {
        self.index.read().contains_key(key) && 
        self.cache_dir.join(key).exists()
    }
    
    /// Get cache statistics
    pub fn get_stats(&self) -> CacheStats {
        let index = self.index.read();
        let total_size_bytes: u64 = index.values().map(|e| e.size_bytes).sum();
        let total_size_mb = total_size_bytes as f64 / 1024.0 / 1024.0;
        
        let hits = *self.hits.read();
        let misses = *self.misses.read();
        let total_requests = hits + misses;
        let hit_rate = if total_requests > 0 {
            hits as f32 / total_requests as f32
        } else {
            0.0
        };
        
        // Find most used entries
        let mut entries: Vec<_> = index.values()
            .map(|e| (e.operation.clone(), e.access_count))
            .collect();
        entries.sort_by(|a, b| b.1.cmp(&a.1));
        
        // Estimate space saved (assuming each computation would take same space)
        let space_saved_mb = entries.iter()
            .map(|(_, count)| {
                if *count > 1 {
                    let entry = index.values().find(|e| e.access_count == *count).unwrap();
                    (entry.size_bytes as f64 / 1024.0 / 1024.0) * (*count - 1) as f64
                } else {
                    0.0
                }
            })
            .sum();
        
        CacheStats {
            total_entries: index.len(),
            total_size_mb,
            hit_rate,
            most_used: entries.into_iter().take(10).collect(),
            space_saved_mb,
        }
    }
    
    /// Clean up old cache entries
    pub fn cleanup_old_entries(&self) -> anyhow::Result<()> {
        let cutoff_date = Utc::now() - Duration::days(self.max_age_days);
        let mut index = self.index.write();
        let mut removed = Vec::new();
        
        for (key, entry) in index.iter() {
            if entry.last_accessed < cutoff_date {
                removed.push(key.clone());
            }
        }
        
        for key in removed {
            index.remove(&key);
            let file_path = self.cache_dir.join(&key);
            fs::remove_file(file_path).ok();
        }
        
        drop(index);
        self.save_index()?;
        
        Ok(())
    }
    
    /// Ensure enough space is available
    async fn ensure_space_available(&self, needed_bytes: u64) -> anyhow::Result<()> {
        let index = self.index.read();
        let current_size: u64 = index.values().map(|e| e.size_bytes).sum();
        let max_size_bytes = (self.max_size_mb * 1024.0 * 1024.0) as u64;
        
        if current_size + needed_bytes > max_size_bytes {
            // Need to evict entries using LRU policy
            let mut entries: Vec<_> = index.values().collect();
            entries.sort_by(|a, b| a.last_accessed.cmp(&b.last_accessed));
            
            let mut space_to_free = (current_size + needed_bytes) - max_size_bytes;
            let mut to_remove = Vec::new();
            
            for entry in entries {
                if space_to_free > 0 {
                    to_remove.push(entry.key.clone());
                    space_to_free = space_to_free.saturating_sub(entry.size_bytes);
                } else {
                    break;
                }
            }
            
            drop(index);
            
            // Remove entries
            let mut index = self.index.write();
            for key in to_remove {
                index.remove(&key);
                let file_path = self.cache_dir.join(&key);
                fs::remove_file(file_path).ok();
            }
        }
        
        Ok(())
    }
    
    /// Save index to disk
    fn save_index(&self) -> anyhow::Result<()> {
        let index_path = self.cache_dir.join("cache_index.json");
        let index = self.index.read();
        let json = serde_json::to_string_pretty(&*index)?;
        fs::write(index_path, json)?;
        Ok(())
    }
    
    /// Clear entire cache
    pub fn clear(&self) -> anyhow::Result<()> {
        let mut index = self.index.write();
        
        // Remove all files
        for key in index.keys() {
            let file_path = self.cache_dir.join(key);
            fs::remove_file(file_path).ok();
        }
        
        index.clear();
        *self.hits.write() = 0;
        *self.misses.write() = 0;
        
        drop(index);
        self.save_index()?;
        
        Ok(())
    }
    
    /// Export cache statistics to file
    pub fn export_stats(&self, output_path: &Path) -> anyhow::Result<()> {
        let stats = self.get_stats();
        let report = format!(
            "Local Computation Cache Report\n\
             ==============================\n\
             Total Entries: {}\n\
             Total Size: {:.2} MB\n\
             Hit Rate: {:.1}%\n\
             Space Saved: {:.2} MB\n\n\
             Most Used Operations:\n{}",
            stats.total_entries,
            stats.total_size_mb,
            stats.hit_rate * 100.0,
            stats.space_saved_mb,
            stats.most_used.iter()
                .map(|(op, count)| format!("  - {}: {} times", op, count))
                .collect::<Vec<_>>()
                .join("\n")
        );
        
        fs::write(output_path, report)?;
        Ok(())
    }
}

/// Helper to cache imputation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedImputationResult {
    pub imputed_data: Vec<Vec<f64>>,
    pub method: String,
    pub n_imputed: usize,
    pub quality_metrics: HashMap<String, f64>,
    pub computation_time_ms: u64,
}

/// Compute hash of data for cache key generation
pub fn hash_data(data: &[Vec<f64>]) -> String {
    let mut hasher = Sha256::new();
    for row in data {
        for &value in row {
            hasher.update(&value.to_le_bytes());
        }
    }
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[tokio::test]
    async fn test_cache_operations() {
        let temp_dir = TempDir::new().unwrap();
        let cache = LocalComputationCache::new(temp_dir.path(), 10.0).unwrap();
        
        // Test data
        let result = CachedImputationResult {
            imputed_data: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            method: "mean".to_string(),
            n_imputed: 2,
            quality_metrics: HashMap::new(),
            computation_time_ms: 100,
        };
        
        let key = LocalComputationCache::generate_key(
            "imputation",
            "test_hash",
            "mean",
            &serde_json::json!({}),
        );
        
        // Store
        cache.store(key.clone(), "imputation".to_string(), &result, HashMap::new())
            .await
            .unwrap();
        
        // Retrieve
        let retrieved: Option<CachedImputationResult> = cache.retrieve(&key).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().method, "mean");
        
        // Check stats
        let stats = cache.get_stats();
        assert_eq!(stats.total_entries, 1);
        assert_eq!(stats.hit_rate, 1.0);
    }
}