use arrow::array::{ArrayRef, Float64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::sync::{mpsc, oneshot};
use uuid::Uuid;
use serde::{Deserialize, Serialize};
use tracing::{error, info};

/// Represents a secure Python operation that can be executed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SafePythonAction {
    // Statistical Methods
    ImputeMean,
    ImputeMedian,
    ImputeMode,
    ImputeForwardFill,
    ImputeBackwardFill,
    ImputeLinearInterpolation,
    ImputeSpline { order: u32 },
    
    // Machine Learning Methods
    ImputeKNN { k: u32 },
    ImputeRandomForest { n_estimators: u32, max_depth: Option<u32> },
    ImputeMICE { max_iter: u32 },
    ImputeXGBoost { n_estimators: u32, learning_rate: f32 },
    
    // Deep Learning Methods
    ImputeAutoencoder { hidden_dims: Vec<u32>, epochs: u32 },
    ImputeGAIN { hidden_dims: Vec<u32>, epochs: u32 },
    ImputeTransformer { model_name: String, context_length: u32 },
    
    // Spatial Methods
    ImputeKriging { variogram_model: String },
    ImputeGNN { hidden_dims: Vec<u32>, num_layers: u32 },
    
    // Ensemble Methods
    ImputeEnsemble { methods: Vec<String>, weights: Option<Vec<f64>> },
    
    // Control Operations
    CancelJob { job_id: Uuid },
    CheckHealth,
}

/// Represents a task to be sent to Python workers
#[derive(Debug, Serialize, Deserialize)]
pub struct PythonTask {
    pub id: Uuid,
    pub action: SafePythonAction,
    pub data: Vec<u8>,  // Serialized Arrow IPC data
    pub metadata: HashMap<String, String>,
}

/// Represents a response from Python workers
#[derive(Debug, Serialize, Deserialize)]
pub struct PythonResponse {
    pub task_id: Uuid,
    pub status: TaskStatus,
    pub result: Option<TaskResult>,
    pub error: Option<TaskError>,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum TaskStatus {
    Success,
    Failed,
    Cancelled,
    InProgress { progress: f32 },
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TaskResult {
    pub data: Vec<u8>,  // Serialized Arrow IPC result
    pub metrics: HashMap<String, f64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TaskError {
    pub error_type: String,
    pub message: String,
    pub traceback: Option<String>,
}

/// Manages a pool of Python worker processes
pub struct PythonWorkerPool {
    workers: Vec<Arc<Mutex<PythonWorker>>>,
    task_sender: mpsc::Sender<(PythonTask, oneshot::Sender<PythonResponse>)>,
    pub num_workers: usize,
}

struct PythonWorker {
    id: usize,
    process: std::process::Child,
    stdin: std::process::ChildStdin,
    stdout: std::process::ChildStdout,
}

impl PythonWorkerPool {
    /// Initialize a new worker pool with specified number of workers
    pub async fn new(num_workers: usize) -> Result<Self, String> {
        let (pool_sender, mut pool_receiver) = mpsc::channel::<(PythonTask, oneshot::Sender<PythonResponse>)>(1000);
        let mut workers = Vec::new();
        
        info!("Initializing Python worker pool with {} workers", num_workers);
        
        for i in 0..num_workers {
            match Self::spawn_worker(i).await {
                Ok(worker) => workers.push(Arc::new(Mutex::new(worker))),
                Err(e) => {
                    error!("Failed to spawn worker {}: {}", i, e);
                    // Clean up already spawned workers
                    for worker in &workers {
                        let mut worker = worker.lock().unwrap();
                        let _ = worker.process.kill();
                    }
                    return Err(format!("Failed to initialize worker pool: {}", e));
                }
            }
        }
        
        // Start the task dispatcher
        let num_workers = workers.len();
        
        // Clone workers into the async block
        let workers_for_dispatcher = workers.clone();
        
        tokio::spawn(async move {
            while let Some((task, response_tx)) = pool_receiver.recv().await {
                // Simple round-robin scheduling for now
                let worker_idx = task.id.as_u128() as usize % num_workers;
                let worker = workers_for_dispatcher[worker_idx].clone();
                
                // Send task to worker
                tokio::spawn(async move {
                    let response = Self::send_task_to_worker(worker, task).await;
                    let _ = response_tx.send(response);
                });
            }
        });
        
        Ok(Self {
            workers,
            task_sender: pool_sender,
            num_workers,
        })
    }
    
    async fn spawn_worker(id: usize) -> Result<PythonWorker, String> {
        use std::io::Write;
        
        // Get the worker script
        let worker_script = include_str!("../../../scripts/airimpute/arrow_worker.py");
        
        // Try python3 first, then python
        let python_cmd = if std::process::Command::new("python3")
            .arg("--version")
            .output()
            .is_ok()
        {
            "python3"
        } else {
            "python"
        };
        
        let mut cmd = std::process::Command::new(python_cmd);
        cmd.arg("-")  // Read script from stdin
            .arg("--worker-id").arg(id.to_string())
            .arg("--ipc-mode").arg("arrow")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
        
        let mut process = cmd.spawn()
            .map_err(|e| format!("Failed to spawn Python worker: {}", e))?;
        
        // Take stdin and stdout handles
        let stdin = process.stdin.take()
            .ok_or_else(|| "Failed to get worker stdin".to_string())?;
        let stdout = process.stdout.take()
            .ok_or_else(|| "Failed to get worker stdout".to_string())?;
        
        // Write the script to stdin and keep it open for communication
        {
            use std::io::Write;
            let mut stdin_ref = &stdin;
            stdin_ref.write_all(worker_script.as_bytes())
                .map_err(|e| format!("Failed to write script to worker stdin: {}", e))?;
            stdin_ref.flush()
                .map_err(|e| format!("Failed to flush worker stdin: {}", e))?;
        }
        
        info!("Spawned Python worker {} with PID {}", id, process.id());
        
        Ok(PythonWorker {
            id,
            process,
            stdin,
            stdout,
        })
    }
    
    /// Send a task to a specific worker and wait for response
    async fn send_task_to_worker(worker: Arc<Mutex<PythonWorker>>, task: PythonTask) -> PythonResponse {
        use std::io::{Write, BufRead, BufReader};
        
        // Serialize task to JSON
        let task_json = match serde_json::to_string(&task) {
            Ok(json) => json,
            Err(e) => {
                return PythonResponse {
                    task_id: task.id,
                    status: TaskStatus::Failed,
                    result: None,
                    error: Some(TaskError {
                        error_type: "SerializationError".to_string(),
                        message: format!("Failed to serialize task: {}", e),
                        traceback: None,
                    }),
                };
            }
        };
        
        // Send task to worker
        let mut worker_guard = worker.lock().unwrap();
        if let Err(e) = writeln!(worker_guard.stdin, "{}", task_json) {
            return PythonResponse {
                task_id: task.id,
                status: TaskStatus::Failed,
                result: None,
                error: Some(TaskError {
                    error_type: "CommunicationError".to_string(),
                    message: format!("Failed to send task to worker: {}", e),
                    traceback: None,
                }),
            };
        }
        
        if let Err(e) = worker_guard.stdin.flush() {
            return PythonResponse {
                task_id: task.id,
                status: TaskStatus::Failed,
                result: None,
                error: Some(TaskError {
                    error_type: "CommunicationError".to_string(),
                    message: format!("Failed to flush stdin: {}", e),
                    traceback: None,
                }),
            };
        }
        
        // Drop the lock to read from stdout  
        drop(worker_guard);
        
        // Read response from worker with a new lock
        let mut worker_guard = worker.lock().unwrap();
        let mut reader = BufReader::new(&mut worker_guard.stdout);
        let mut response_line = String::new();
        
        match reader.read_line(&mut response_line) {
            Ok(0) => {
                // EOF
                PythonResponse {
                    task_id: task.id,
                    status: TaskStatus::Failed,
                    result: None,
                    error: Some(TaskError {
                        error_type: "WorkerDied".to_string(),
                        message: "Worker process terminated unexpectedly".to_string(),
                        traceback: None,
                    }),
                }
            }
            Ok(_) => {
                // Parse response
                match serde_json::from_str(&response_line) {
                    Ok(response) => response,
                    Err(e) => PythonResponse {
                        task_id: task.id,
                        status: TaskStatus::Failed,
                        result: None,
                        error: Some(TaskError {
                            error_type: "DeserializationError".to_string(),
                            message: format!("Failed to parse worker response: {}", e),
                            traceback: None,
                        }),
                    },
                }
            }
            Err(e) => {
                PythonResponse {
                    task_id: task.id,
                    status: TaskStatus::Failed,
                    result: None,
                    error: Some(TaskError {
                        error_type: "CommunicationError".to_string(),
                        message: format!("Failed to read from worker: {}", e),
                        traceback: None,
                    }),
                }
            }
        }
    }
    
    /// Execute a task on the worker pool
    pub async fn execute(&self, task: PythonTask) -> Result<PythonResponse, String> {
        let (response_tx, response_rx) = oneshot::channel();
        
        self.task_sender.send((task, response_tx)).await
            .map_err(|e| format!("Failed to send task to pool: {}", e))?;
        
        response_rx.await
            .map_err(|e| format!("Failed to receive response: {}", e))
    }
    
    /// Shutdown all workers gracefully
    pub async fn shutdown(&mut self) -> Result<(), String> {
        info!("Shutting down Python worker pool");
        
        for worker in &self.workers {
            let mut worker_guard = worker.lock().unwrap();
            
            // Force kill the process
            let _ = worker_guard.process.kill();
        }
        
        Ok(())
    }
}

/// Convert ndarray to Arrow RecordBatch for zero-copy transfer
pub fn ndarray_to_arrow(
    data: &ndarray::Array2<f64>,
    column_names: &[String],
) -> Result<RecordBatch, String> {
    if data.ncols() != column_names.len() {
        return Err(format!(
            "Column count mismatch: data has {} columns, but {} names provided",
            data.ncols(),
            column_names.len()
        ));
    }
    
    let mut columns: Vec<ArrayRef> = Vec::new();
    let mut fields: Vec<Field> = Vec::new();
    
    for (col_idx, col_name) in column_names.iter().enumerate() {
        let col_data: Vec<f64> = data.column(col_idx).to_vec();
        let array = Arc::new(Float64Array::from(col_data)) as ArrayRef;
        columns.push(array);
        fields.push(Field::new(col_name, DataType::Float64, true));
    }
    
    let schema = Arc::new(Schema::new(fields));
    RecordBatch::try_new(schema, columns)
        .map_err(|e| format!("Failed to create RecordBatch: {}", e))
}

/// Convert Arrow RecordBatch back to ndarray
pub fn arrow_to_ndarray(batch: &RecordBatch) -> Result<ndarray::Array2<f64>, String> {
    let num_rows = batch.num_rows();
    let num_cols = batch.num_columns();
    
    let mut data = ndarray::Array2::<f64>::zeros((num_rows, num_cols));
    
    for col_idx in 0..num_cols {
        let column = batch.column(col_idx);
        let float_array = column
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| format!("Column {} is not Float64Array", col_idx))?;
        
        for (row_idx, value) in float_array.iter().enumerate() {
            data[[row_idx, col_idx]] = value.unwrap_or(f64::NAN);
        }
    }
    
    Ok(data)
}

/// Serialize RecordBatch to Arrow IPC format for inter-process communication
pub fn serialize_record_batch(batch: &RecordBatch) -> Result<Vec<u8>, String> {
    use arrow::ipc::writer::StreamWriter;
    
    let mut buffer = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buffer, &batch.schema())
            .map_err(|e| format!("Failed to create StreamWriter: {}", e))?;
        
        writer.write(batch)
            .map_err(|e| format!("Failed to write batch: {}", e))?;
        
        writer.finish()
            .map_err(|e| format!("Failed to finish writing: {}", e))?;
    }
    
    Ok(buffer)
}

/// Deserialize RecordBatch from Arrow IPC format
pub fn deserialize_record_batch(data: &[u8]) -> Result<RecordBatch, String> {
    use arrow::ipc::reader::StreamReader;
    use std::io::Cursor;
    
    let cursor = Cursor::new(data);
    let mut reader = StreamReader::try_new(cursor, None)
        .map_err(|e| format!("Failed to create StreamReader: {}", e))?;
    
    match reader.next() {
        Some(Ok(batch)) => Ok(batch),
        Some(Err(e)) => Err(format!("Failed to read batch: {}", e)),
        None => Err("No batch found in stream".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;
    
    #[test]
    fn test_ndarray_to_arrow_conversion() {
        let data = arr2(&[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]);
        
        let column_names = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        
        let batch = ndarray_to_arrow(&data, &column_names).unwrap();
        assert_eq!(batch.num_rows(), 3);
        assert_eq!(batch.num_columns(), 3);
        
        let data_back = arrow_to_ndarray(&batch).unwrap();
        assert_eq!(data, data_back);
    }
    
    #[test]
    fn test_safe_python_action_serialization() {
        let action = SafePythonAction::ImputeKNN { k: 5 };
        let serialized = serde_json::to_string(&action).unwrap();
        let deserialized: SafePythonAction = serde_json::from_str(&serialized).unwrap();
        
        match deserialized {
            SafePythonAction::ImputeKNN { k } => assert_eq!(k, 5),
            _ => panic!("Unexpected action type"),
        }
    }
}