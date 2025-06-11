use lru::LruCache;
use std::sync::Mutex;
use std::num::NonZeroUsize;

/// Simple LRU cache manager
pub struct CacheManager {
    cache: Mutex<LruCache<String, Vec<u8>>>,
    max_size: usize,
}

impl CacheManager {
    pub fn new(max_size: usize) -> Self {
        let cache = LruCache::new(NonZeroUsize::new(1000).unwrap());
        Self {
            cache: Mutex::new(cache),
            max_size,
        }
    }
    
    pub fn insert(&self, key: String, value: Vec<u8>) {
        if value.len() <= self.max_size {
            self.cache.lock().unwrap().put(key, value);
        }
    }
    
    pub fn get(&self, key: &str) -> Option<Vec<u8>> {
        self.cache.lock().unwrap().get(key).cloned()
    }
}