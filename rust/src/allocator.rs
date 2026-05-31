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
    pub recorded_streams: Vec<u32>,
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
        // Updated to support cross-stream/cross-tag reuse (Stream-Aware Allocation)
        for block in blocks.values_mut() {
            if block.is_idle && block.size >= size {
                // blockFree: block is safe for reuse since all recorded streams are completed in our CPU model.
                println!("🚀 [StreamAwareAllocator] blockFree: Reusing block address 0x{:x} from stream {} (now stream {})", block.ptr, block.stream_id, stream_id);
                block.is_idle = false;
                block.stream_id = stream_id;
                block.tag = tag.clone();
                block.recorded_streams.clear();
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
            recorded_streams: Vec::new(),
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
            println!("🚀 [StreamAwareAllocator] free: User-level deletion of pointer address 0x{:x} (stream {})", ptr, stream_id);
        }
        
        Ok(())
    }

    pub fn record_stream(&self, ptr: usize, stream_id: u32) -> PyResult<()> {
        let mut blocks = self.blocks.lock().unwrap();
        if let Some(block) = blocks.get_mut(&ptr) {
            if !block.recorded_streams.contains(&stream_id) {
                block.recorded_streams.push(stream_id);
                println!("🚀 [StreamAwareAllocator] record_stream: Tracking stream {} dependency on block 0x{:x}", stream_id, ptr);
            }
        }
        Ok(())
    }

    pub fn cuda_free(&self, ptr: usize) -> PyResult<()> {
        let mut blocks = self.blocks.lock().unwrap();
        if blocks.remove(&ptr).is_some() {
            println!("🚀 [StreamAwareAllocator] cudaFree(): API-level release of GPU memory for address 0x{:x}", ptr);
        }
        
        // Proactive metadata reconciliation to avoid OOM latency spikes
        blocks.retain(|_, block| {
            if block.is_idle && block.recorded_streams.is_empty() {
                println!("🚀 [StreamAwareAllocator] Proactive Reclaim: Freeing address 0x{:x} from heap", block.ptr);
                false
            } else {
                true
            }
        });
        Ok(())
    }

    pub fn reclaim_idle_blocks(&self) -> PyResult<()> {
        let mut blocks = self.blocks.lock().unwrap();
        blocks.retain(|_, block| {
            if block.is_idle && block.recorded_streams.is_empty() {
                println!("🚀 [StreamAwareAllocator] Proactive Reclaim: Freeing address 0x{:x} from heap", block.ptr);
                false
            } else {
                true
            }
        });
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
