// Local file manager with integrity checking and versioning
use std::path::{Path, PathBuf};
use std::fs::{self, File};
use std::io::{Read, BufReader};
use serde::{Serialize, Deserialize};
use sha2::{Sha256, Digest};
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use anyhow::Result;
use zip::write::FileOptions;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManagedFile {
    pub id: String,
    pub original_name: String,
    pub stored_name: String,
    pub file_type: FileType,
    pub size_bytes: u64,
    pub hash: String,
    pub created_at: DateTime<Utc>,
    pub last_accessed: DateTime<Utc>,
    pub access_count: u32,
    pub metadata: FileMetadata,
    pub versions: Vec<FileVersion>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FileType {
    Dataset,
    ImputationResult,
    Report,
    Configuration,
    Script,
    Export,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FileMetadata {
    pub rows: Option<usize>,
    pub columns: Option<usize>,
    pub format: String,
    pub compression: Option<String>,
    pub encoding: String,
    pub custom: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileVersion {
    pub version: u32,
    pub hash: String,
    pub size_bytes: u64,
    pub created_at: DateTime<Utc>,
    pub description: String,
}

pub struct LocalFileManager {
    base_dir: PathBuf,
    index: HashMap<String, ManagedFile>,
    max_file_size_mb: f64,
    enable_versioning: bool,
    enable_compression: bool,
}

impl LocalFileManager {
    pub fn new(app_data_dir: &Path) -> Result<Self> {
        let base_dir = app_data_dir.join("managed_files");
        fs::create_dir_all(&base_dir)?;
        
        // Load index
        let index_path = base_dir.join("file_index.json");
        let index = if index_path.exists() {
            let content = fs::read_to_string(&index_path)?;
            serde_json::from_str(&content).unwrap_or_default()
        } else {
            HashMap::new()
        };
        
        Ok(Self {
            base_dir,
            index,
            max_file_size_mb: 1000.0, // 1GB limit
            enable_versioning: true,
            enable_compression: true,
        })
    }
    
    /// Import file into managed storage
    pub fn import_file(&mut self, 
                      source_path: &Path, 
                      file_type: FileType,
                      metadata: Option<FileMetadata>) -> Result<String> {
        // Check file size
        let file_meta = fs::metadata(source_path)?;
        let size_mb = file_meta.len() as f64 / 1024.0 / 1024.0;
        
        if size_mb > self.max_file_size_mb {
            anyhow::bail!("File too large: {:.1}MB (max: {:.1}MB)", 
                         size_mb, self.max_file_size_mb);
        }
        
        // Generate file ID and hash
        let file_id = uuid::Uuid::new_v4().to_string();
        let file_hash = self.calculate_file_hash(source_path)?;
        
        // Check for duplicates
        if let Some(existing) = self.find_by_hash(&file_hash) {
            return Ok(existing.id.clone());
        }
        
        // Determine storage name
        let original_name = source_path.file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        let stored_name = format!("{}.{}", file_id, 
            source_path.extension().unwrap_or_default().to_string_lossy());
        
        // Create type-specific directory
        let type_dir = self.base_dir.join(self.type_to_dir(&file_type));
        fs::create_dir_all(&type_dir)?;
        
        let dest_path = type_dir.join(&stored_name);
        
        // Copy or compress file
        if self.enable_compression && self.should_compress(&original_name) {
            self.compress_file(source_path, &dest_path)?;
        } else {
            fs::copy(source_path, &dest_path)?;
        }
        
        // Create metadata
        let metadata = metadata.unwrap_or_else(|| {
            self.detect_metadata(source_path).unwrap_or_default()
        });
        
        // Create file entry
        let managed_file = ManagedFile {
            id: file_id.clone(),
            original_name,
            stored_name,
            file_type,
            size_bytes: file_meta.len(),
            hash: file_hash.clone(),
            created_at: Utc::now(),
            last_accessed: Utc::now(),
            access_count: 0,
            metadata,
            versions: vec![FileVersion {
                version: 1,
                hash: file_hash.clone(),
                size_bytes: file_meta.len(),
                created_at: Utc::now(),
                description: "Initial version".to_string(),
            }],
        };
        
        // Add to index
        self.index.insert(file_id.clone(), managed_file);
        self.save_index()?;
        
        Ok(file_id)
    }
    
    /// Export file from managed storage
    pub fn export_file(&mut self, file_id: &str, dest_path: &Path) -> Result<()> {
        // Extract necessary data to avoid borrow conflicts
        let (file_type, stored_name, file_hash) = {
            let file = self.index.get_mut(file_id)
                .ok_or_else(|| anyhow::anyhow!("File not found: {}", file_id))?;
            
            // Update access stats
            file.last_accessed = Utc::now();
            file.access_count += 1;
            
            // Clone the data we need
            (file.file_type.clone(), file.stored_name.clone(), file.hash.clone())
        };
        
        let source_path = self.base_dir
            .join(self.type_to_dir(&file_type))
            .join(&stored_name);
        
        if !source_path.exists() {
            anyhow::bail!("File data missing: {}", file_id);
        }
        
        // Copy or decompress
        if self.is_compressed(&source_path) {
            self.decompress_file(&source_path, dest_path)?;
        } else {
            fs::copy(&source_path, dest_path)?;
        }
        
        // Verify integrity
        let exported_hash = self.calculate_file_hash(dest_path)?;
        if exported_hash != file_hash {
            anyhow::bail!("File integrity check failed");
        }
        
        self.save_index()?;
        
        Ok(())
    }
    
    /// Create new version of file
    pub fn create_version(&mut self, 
                         file_id: &str, 
                         new_file_path: &Path,
                         description: String) -> Result<u32> {
        if !self.enable_versioning {
            anyhow::bail!("Versioning is disabled");
        }
        
        // Calculate new file properties without mutating self
        let new_hash = self.calculate_file_hash(new_file_path)?;
        let new_meta = fs::metadata(new_file_path)?;
        
        // Read from self and clone needed data in a limited scope
        let (file_id_clone, original_name, file_type, new_version) = {
            let file = self.index.get(file_id) // Use immutable get first
                .ok_or_else(|| anyhow::anyhow!("File not found: {}", file_id))?;
            
            // If the content is unchanged, return the current version number
            if new_hash == file.hash {
                // It's safe to unwrap, as a file must have at least one version
                return Ok(file.versions.last().unwrap().version);
            }

            // Clone data needed for operations outside this scope
            (
                file.id.clone(),
                file.original_name.clone(),
                file.file_type.clone(),
                file.versions.len() as u32 + 1
            )
        };
        
        // Perform file system operations using cloned data
        let versioned_name = format!("{}_v{}.{}", 
            file_id_clone, 
            new_version,
            Path::new(&original_name).extension()
                .unwrap_or_default().to_string_lossy());
        
        let type_dir = self.base_dir.join(self.type_to_dir(&file_type));
        let dest_path = type_dir.join(&versioned_name);
        
        if self.enable_compression && self.should_compress(&original_name) {
            self.compress_file(new_file_path, &dest_path)?;
        } else {
            fs::copy(new_file_path, &dest_path)?;
        }
        
        // Mutate the file entry in self.index
        let file = self.index.get_mut(file_id)
            .expect("File should exist as we checked it before");

        file.versions.push(FileVersion {
            version: new_version,
            hash: new_hash.clone(),
            size_bytes: new_meta.len(),
            created_at: Utc::now(),
            description,
        });
        
        // Update the main file entry to point to the new version
        file.hash = new_hash;
        file.size_bytes = new_meta.len();
        file.stored_name = versioned_name;
        
        self.save_index()?;
        
        Ok(new_version)
    }
    
    /// Get file information
    pub fn get_file_info(&self, file_id: &str) -> Option<&ManagedFile> {
        self.index.get(file_id)
    }
    
    /// List files by type
    pub fn list_files(&self, file_type: Option<FileType>) -> Vec<&ManagedFile> {
        self.index.values()
            .filter(|f| file_type.as_ref().map_or(true, |t| 
                std::mem::discriminant(&f.file_type) == std::mem::discriminant(t)))
            .collect()
    }
    
    /// Search files
    pub fn search_files(&self, query: &str) -> Vec<&ManagedFile> {
        let query_lower = query.to_lowercase();
        
        self.index.values()
            .filter(|f| {
                f.original_name.to_lowercase().contains(&query_lower) ||
                f.metadata.custom.values()
                    .any(|v| v.to_string().to_lowercase().contains(&query_lower))
            })
            .collect()
    }
    
    /// Clean up old versions
    pub fn cleanup_old_versions(&mut self, keep_versions: usize) -> Result<u64> {
        let mut freed_bytes = 0u64;
        let mut paths_to_delete: Vec<PathBuf> = Vec::new();

        // Collect IDs of files that need cleanup to avoid borrow conflicts
        let file_ids_to_process: Vec<String> = self.index.values()
            .filter(|f| f.versions.len() > keep_versions)
            .map(|f| f.id.clone())
            .collect();

        // Step 1: Collect paths of version files to delete
        for file_id in &file_ids_to_process {
            let file = self.index.get(file_id).unwrap(); // We know it exists
            
            let mut sorted_versions = file.versions.clone();
            sorted_versions.sort_by_key(|v| v.version);

            let to_remove_count = sorted_versions.len().saturating_sub(keep_versions);
            for version_to_remove in sorted_versions.iter().take(to_remove_count) {
                let versioned_name = format!("{}_v{}.{}", 
                    file.id, 
                    version_to_remove.version,
                    Path::new(&file.original_name).extension()
                        .unwrap_or_default().to_string_lossy());
                
                let type_dir = self.base_dir.join(self.type_to_dir(&file.file_type));
                paths_to_delete.push(type_dir.join(versioned_name));
            }
        }

        // Step 2: Delete the physical files from disk
        for path in paths_to_delete {
            if path.exists() {
                if let Ok(meta) = fs::metadata(&path) {
                    if fs::remove_file(path).is_ok() {
                        freed_bytes += meta.len();
                    }
                }
            }
        }

        // Step 3: Mutate the index to remove old version entries
        for file_id in file_ids_to_process {
            if let Some(file) = self.index.get_mut(&file_id) {
                if file.versions.len() > keep_versions {
                    file.versions.sort_by_key(|v| v.version);
                    let to_remove_count = file.versions.len() - keep_versions;
                    // drain efficiently removes the items in place
                    file.versions.drain(..to_remove_count);
                }
            }
        }
        
        self.save_index()?;
        
        Ok(freed_bytes)
    }
    
    /// Calculate file hash
    fn calculate_file_hash(&self, path: &Path) -> Result<String> {
        let mut file = File::open(path)?;
        let mut hasher = Sha256::new();
        let mut buffer = [0; 8192];
        
        loop {
            let n = file.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            hasher.update(&buffer[..n]);
        }
        
        Ok(format!("{:x}", hasher.finalize()))
    }
    
    /// Find file by hash
    fn find_by_hash(&self, hash: &str) -> Option<&ManagedFile> {
        self.index.values().find(|f| f.hash == hash)
    }
    
    /// Get directory for file type
    fn type_to_dir(&self, file_type: &FileType) -> &'static str {
        match file_type {
            FileType::Dataset => "datasets",
            FileType::ImputationResult => "results",
            FileType::Report => "reports",
            FileType::Configuration => "configs",
            FileType::Script => "scripts",
            FileType::Export => "exports",
        }
    }
    
    /// Check if file should be compressed
    fn should_compress(&self, filename: &str) -> bool {
        let ext = Path::new(filename).extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        
        // Don't compress already compressed formats
        !matches!(ext.as_str(), "zip" | "gz" | "bz2" | "xz" | "7z" | "rar" | 
                               "jpg" | "jpeg" | "png" | "mp4" | "avi" | "parquet")
    }
    
    /// Compress file
    fn compress_file(&self, source: &Path, dest: &Path) -> Result<()> {
        use zip::ZipWriter;
        
        let file = File::create(dest)?;
        let mut zip = ZipWriter::new(file);
        
        let options = FileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated)
            .unix_permissions(0o644);
        
        let filename = source.file_name()
            .unwrap_or_default()
            .to_string_lossy();
        
        zip.start_file(filename, options)?;
        
        let mut source_file = File::open(source)?;
        std::io::copy(&mut source_file, &mut zip)?;
        
        zip.finish()?;
        
        Ok(())
    }
    
    /// Decompress file
    fn decompress_file(&self, source: &Path, dest: &Path) -> Result<()> {
        use zip::ZipArchive;
        
        let file = File::open(source)?;
        let mut archive = ZipArchive::new(file)?;
        
        // Extract first file
        let mut file = archive.by_index(0)?;
        let mut dest_file = File::create(dest)?;
        
        std::io::copy(&mut file, &mut dest_file)?;
        
        Ok(())
    }
    
    /// Check if file is compressed
    fn is_compressed(&self, path: &Path) -> bool {
        path.extension()
            .and_then(|e| e.to_str())
            .map(|e| e == "zip")
            .unwrap_or(false)
    }
    
    /// Detect file metadata
    fn detect_metadata(&self, path: &Path) -> Result<FileMetadata> {
        let ext = path.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        
        let mut metadata = FileMetadata {
            rows: None,
            columns: None,
            format: ext.clone(),
            compression: None,
            encoding: "UTF-8".to_string(),
            custom: HashMap::new(),
        };
        
        // Try to detect CSV properties
        if ext == "csv" {
            if let Ok(file) = File::open(path) {
                let reader = BufReader::new(file);
                let mut csv_reader = csv::Reader::from_reader(reader);
                
                if let Ok(headers) = csv_reader.headers() {
                    metadata.columns = Some(headers.len());
                }
                
                // Count rows (limited sample)
                let row_count = csv_reader.records().take(1000).count();
                if row_count == 1000 {
                    metadata.rows = None; // Too many to count efficiently
                } else {
                    metadata.rows = Some(row_count);
                }
            }
        }
        
        Ok(metadata)
    }
    
    /// Save index to disk
    fn save_index(&self) -> Result<()> {
        let index_path = self.base_dir.join("file_index.json");
        let json = serde_json::to_string_pretty(&self.index)?;
        fs::write(index_path, json)?;
        Ok(())
    }
    
    /// Get storage statistics
    pub fn get_storage_stats(&self) -> StorageStats {
        let mut stats = StorageStats::default();
        
        for file in self.index.values() {
            stats.total_files += 1;
            stats.total_size_bytes += file.size_bytes;
            stats.total_versions += file.versions.len();
            
            match file.file_type {
                FileType::Dataset => stats.datasets += 1,
                FileType::ImputationResult => stats.results += 1,
                FileType::Report => stats.reports += 1,
                _ => {}
            }
        }
        
        stats
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct StorageStats {
    pub total_files: usize,
    pub total_size_bytes: u64,
    pub total_versions: usize,
    pub datasets: usize,
    pub results: usize,
    pub reports: usize,
}

impl StorageStats {
    pub fn total_size_mb(&self) -> f64 {
        self.total_size_bytes as f64 / 1024.0 / 1024.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[test]
    fn test_file_import() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = LocalFileManager::new(temp_dir.path()).unwrap();
        
        // Create test file
        let test_file = temp_dir.path().join("test.csv");
        fs::write(&test_file, "col1,col2\n1,2\n3,4").unwrap();
        
        // Import file
        let file_id = manager.import_file(
            &test_file,
            FileType::Dataset,
            None
        ).unwrap();
        
        // Check file info
        let info = manager.get_file_info(&file_id).unwrap();
        assert_eq!(info.original_name, "test.csv");
        assert_eq!(info.file_type, FileType::Dataset);
    }
}