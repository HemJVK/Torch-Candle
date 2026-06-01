use std::sync::atomic::{AtomicUsize, Ordering};
use pyo3::prelude::*;
use std::sync::Arc;
use parking_lot::Mutex;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[pyclass]
pub struct TaskMetadata {
    #[pyo3(get)]
    pub op_code: u32,
    #[pyo3(get)]
    pub device_id: u32,
    #[pyo3(get)]
    pub input_size: u64,
    #[pyo3(get)]
    pub output_size: u64,
    #[pyo3(get)]
    pub payload: [u8; 256],
}

impl Default for TaskMetadata {
    fn default() -> Self {
        Self {
            op_code: 0,
            device_id: 0,
            input_size: 0,
            output_size: 0,
            payload: [0; 256],
        }
    }
}

// 128-byte cache alignment to prevent false sharing and MESI invalidations
#[repr(align(128))]
pub struct CacheAlignedAtomicUsize {
    pub val: AtomicUsize,
}

#[repr(C)]
pub struct SPSCRingBufferLayout {
    pub head: CacheAlignedAtomicUsize,
    pub tail: CacheAlignedAtomicUsize,
    pub buffer: [TaskMetadata; 1024],
}

#[pyclass]
pub struct SPSCRingBuffer {
    raw_ptr: *mut SPSCRingBufferLayout,
    is_owner: bool,
    condvar: Arc<std::sync::Condvar>,
    condvar_mutex: Arc<std::sync::Mutex<()>>,
}

unsafe impl Send for SPSCRingBuffer {}
unsafe impl Sync for SPSCRingBuffer {}

impl Clone for SPSCRingBuffer {
    fn clone(&self) -> Self {
        Self {
            raw_ptr: self.raw_ptr,
            is_owner: false,
            condvar: self.condvar.clone(),
            condvar_mutex: self.condvar_mutex.clone(),
        }
    }
}

#[pymethods]
impl SPSCRingBuffer {
    #[new]
    pub fn new() -> Self {
        let layout = Box::new(SPSCRingBufferLayout {
            head: CacheAlignedAtomicUsize { val: AtomicUsize::new(0) },
            tail: CacheAlignedAtomicUsize { val: AtomicUsize::new(0) },
            buffer: [TaskMetadata::default(); 1024],
        });
        let raw_ptr = Box::into_raw(layout);
        Self {
            raw_ptr,
            is_owner: true,
            condvar: Arc::new(std::sync::Condvar::new()),
            condvar_mutex: Arc::new(std::sync::Mutex::new(())),
        }
    }

    pub fn push(&self, op_code: u32, device_id: u32, payload_bytes: Vec<u8>) -> PyResult<()> {
        let layout = unsafe { &mut *self.raw_ptr };
        let head = layout.head.val.load(Ordering::Relaxed);
        let tail = layout.tail.val.load(Ordering::Acquire);
        
        if head.wrapping_sub(tail) >= 1024 {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Ring buffer is full"));
        }
        
        let index = head % 1024;
        let mut payload = [0u8; 256];
        let copy_len = payload_bytes.len().min(256);
        payload[..copy_len].copy_from_slice(&payload_bytes[..copy_len]);
        
        layout.buffer[index] = TaskMetadata {
            op_code,
            device_id,
            input_size: 0,
            output_size: 0,
            payload,
        };
        
        layout.head.val.store(head.wrapping_add(1), Ordering::Release);
        
        self.condvar.notify_all();
        
        Ok(())
    }

    pub fn pop(&self) -> PyResult<Option<TaskMetadata>> {
        let layout = unsafe { &mut *self.raw_ptr };
        let head = layout.head.val.load(Ordering::Acquire);
        let tail = layout.tail.val.load(Ordering::Relaxed);
        
        if tail == head {
            return Ok(None);
        }
        
        let index = tail % 1024;
        let task = layout.buffer[index];
        
        layout.tail.val.store(tail.wrapping_add(1), Ordering::Release);
        Ok(Some(task))
    }

    pub fn wait_and_pop(&self, py: Python<'_>) -> PyResult<TaskMetadata> {
        let layout = unsafe { &mut *self.raw_ptr };
        let tail = layout.tail.val.load(Ordering::Relaxed);
        
        py.allow_threads(|| {
            let start = std::time::Instant::now();
            while layout.head.val.load(Ordering::Acquire) == tail {
                let elapsed = start.elapsed();
                if elapsed.as_micros() < 50 {
                    std::hint::spin_loop();
                } else if elapsed.as_micros() < 1000 {
                    std::thread::yield_now();
                } else {
                    let mut guard = self.condvar_mutex.lock().unwrap();
                    while layout.head.val.load(Ordering::Acquire) == tail {
                        guard = self.condvar.wait(guard).unwrap();
                    }
                    break;
                }
            }
            Ok::<(), PyErr>(())
        })?;
        
        let index = tail % 1024;
        let task = layout.buffer[index];
        layout.tail.val.store(tail.wrapping_add(1), Ordering::Release);
        Ok(task)
    }
}

impl Drop for SPSCRingBuffer {
    fn drop(&mut self) {
        if self.is_owner {
            unsafe {
                let _ = Box::from_raw(self.raw_ptr);
            }
        }
    }
}
