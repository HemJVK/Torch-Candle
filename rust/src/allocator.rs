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

use std::sync::atomic::{AtomicUsize, Ordering};

#[pyclass]
pub struct StreamAwareAllocator {
    blocks: Mutex<HashMap<usize, AllocationBlock>>,
    event_queue: Mutex<VecDeque<StreamEvent>>,
    next_ptr: Mutex<usize>,
    next_event_id: Mutex<u32>,
    pub stream_head: AtomicUsize,
    _pad: [u8; 128],
    pub stream_tail: AtomicUsize,
    waiter_thread: Mutex<Option<std::thread::Thread>>,
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
            stream_head: AtomicUsize::new(0),
            _pad: [0u8; 128],
            stream_tail: AtomicUsize::new(0),
            waiter_thread: Mutex::new(None),
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
        if let Some(block) = blocks.get_mut(&ptr) {
            block.is_idle = true;
            println!("🚀 [StreamAwareAllocator] cudaFree(): API-level logical release of block 0x{:x}", ptr);
        }
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
            std::hint::spin_loop();
        }
        Ok(())
    }

    pub fn increment_stream_head(&self) -> PyResult<()> {
        self.stream_head.fetch_add(1, Ordering::Release);
        
        // Unpark waiter thread if it was parked (Hybrid Wait Strategy Integration)
        if let Ok(mut guard) = self.waiter_thread.lock() {
            if let Some(thread) = guard.take() {
                thread.unpark();
            }
        }
        
        Ok(())
    }

    pub fn get_stream_head(&self) -> PyResult<usize> {
        Ok(self.stream_head.load(Ordering::Acquire))
    }

    pub fn increment_stream_tail(&self) -> PyResult<()> {
        self.stream_tail.fetch_add(1, Ordering::Release);
        Ok(())
    }

    pub fn wait_for_stream_completion(&self, target: usize) -> PyResult<()> {
        let start = std::time::Instant::now();
        loop {
            let current = self.stream_head.load(Ordering::Acquire);
            if current >= target {
                break;
            }
            
            let elapsed = start.elapsed();
            if elapsed.as_micros() < 50 {
                // 1. Busy spinning (< 50µs)
                std::hint::spin_loop();
            } else if elapsed.as_micros() < 500 {
                // 2. Yielding (< 500µs)
                std::thread::yield_now();
            } else {
                // 3. Thread parking/Futex sleep to prevent burning CPU
                if let Ok(mut guard) = self.waiter_thread.lock() {
                    *guard = Some(std::thread::current());
                }
                
                // Double check before parking to avoid race condition
                let check = self.stream_head.load(Ordering::Acquire);
                if check >= target {
                    break;
                }
                
                std::thread::park_timeout(std::time::Duration::from_millis(10));
            }
        }
        Ok(())
    }
}
