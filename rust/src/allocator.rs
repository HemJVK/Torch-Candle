use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use pyo3::prelude::*;

#[pyclass]
#[derive(Clone, Debug)]
pub struct StreamEvent {
    #[pyo3(get)]
    pub stream_id: u32,
    #[pyo3(get)]
    pub event_id: u32,
    #[pyo3(get)]
    pub is_complete: bool,
}

#[pymethods]
impl StreamEvent {
    #[new]
    pub fn new(stream_id: u32, event_id: u32) -> Self {
        Self { stream_id, event_id, is_complete: false }
    }

    pub fn query(&self) -> bool {
        // GPU event completion check. Natively true in CPU fallback.
        true
    }
}

#[derive(Clone, Debug)]
pub struct AllocationBlock {
    pub ptr: usize,
    pub size: usize,
    pub stream_id: u32,
    pub is_idle: bool,
    pub tag: String,
}

#[pyclass]
pub struct StreamAwareAllocator {
    blocks: Mutex<HashMap<usize, AllocationBlock>>,
    event_queue: Mutex<VecDeque<StreamEvent>>,
    next_ptr: Mutex<usize>,
    next_event_id: Mutex<u32>,
}

#[pymethods]
impl StreamAwareAllocator {
    #[new]
    pub fn new() -> Self {
        Self {
            blocks: Mutex::new(HashMap::new()),
            event_queue: Mutex::new(VecDeque::new()),
            next_ptr: Mutex::new(1000000),
            next_event_id: Mutex::new(1),
        }
    }

    pub fn allocate(&self, size: usize, stream_id: u32, tag: String) -> PyResult<usize> {
        let mut blocks = self.blocks.lock().unwrap();
        for block in blocks.values_mut() {
            if block.is_idle && block.size >= size && block.stream_id == stream_id {
                block.is_idle = false;
                block.tag = tag.clone();
                return Ok(block.ptr);
            }
        }
        
        let mut next_ptr = self.next_ptr.lock().unwrap();
        let ptr = *next_ptr;
        *next_ptr += size;
        
        blocks.insert(ptr, AllocationBlock {
            ptr,
            size,
            stream_id,
            is_idle: false,
            tag,
        });
        
        Ok(ptr)
    }

    pub fn free(&self, ptr: usize, stream_id: u32) -> PyResult<()> {
        let mut event_queue = self.event_queue.lock().unwrap();
        let mut next_event_id = self.next_event_id.lock().unwrap();
        
        let event_id = *next_event_id;
        *next_event_id += 1;
        
        let event = StreamEvent {
            stream_id,
            event_id,
            is_complete: false,
        };
        event_queue.push_back(event);
        
        let mut blocks = self.blocks.lock().unwrap();
        if let Some(block) = blocks.get_mut(&ptr) {
            block.is_idle = true; 
        }
        
        Ok(())
    }

    pub fn record_event(&self, stream_id: u32) -> PyResult<StreamEvent> {
        let mut next_event_id = self.next_event_id.lock().unwrap();
        let event_id = *next_event_id;
        *next_event_id += 1;
        Ok(StreamEvent {
            stream_id,
            event_id,
            is_complete: true,
        })
    }

    pub fn wait_event(&self, _comm_stream_id: u32, event: StreamEvent) -> PyResult<()> {
        while !event.query() {
            std::thread::yield_now();
        }
        Ok(())
    }
}
