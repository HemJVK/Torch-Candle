use std::sync::atomic::{AtomicUsize, Ordering};
use pyo3::prelude::*;

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
}

unsafe impl Send for SPSCRingBuffer {}
unsafe impl Sync for SPSCRingBuffer {}

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
        Self { raw_ptr, is_owner: true }
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

    pub fn wait_and_pop(&self) -> PyResult<TaskMetadata> {
        let layout = unsafe { &mut *self.raw_ptr };
        let start = std::time::Instant::now();
        let tail = layout.tail.val.load(Ordering::Relaxed);
        
        loop {
            let head = layout.head.val.load(Ordering::Acquire);
            if tail != head {
                let index = tail % 1024;
                let task = layout.buffer[index];
                layout.tail.val.store(tail.wrapping_add(1), Ordering::Release);
                return Ok(task);
            }
            
            let elapsed = start.elapsed();
            if elapsed.as_micros() < 50 {
                std::hint::spin_loop();
            } else if elapsed.as_micros() < 500 {
                std::thread::yield_now();
            } else {
                std::thread::sleep(std::time::Duration::from_micros(100));
            }
        }
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
