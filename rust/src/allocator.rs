use std::collections::{HashMap, VecDeque};
use std::sync::{Mutex, Condvar};
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
        // Check actual event completion state.
        // On real GPU hardware, this would query cudaEventQuery/hipEventQuery.
        self.is_complete
    }

    pub fn mark_complete(&mut self) {
        self.is_complete = true;
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
    #[allow(dead_code)]
    event_queue: Mutex<VecDeque<StreamEvent>>,
    free_queue: Mutex<VecDeque<(StreamEvent, usize)>>,
    next_ptr: Mutex<usize>,
    next_event_id: Mutex<u32>,
    pub stream_head: AtomicUsize,
    _pad: [u8; 128],
    pub stream_tail: AtomicUsize,
    condvar: Condvar,
    condvar_mutex: Mutex<()>,
}

impl StreamAwareAllocator {
    fn process_delayed_frees_internal(&self, blocks: &mut HashMap<usize, AllocationBlock>) {
        let mut free_queue = self.free_queue.lock().unwrap();
        let mut remaining = VecDeque::new();
        while let Some((event, ptr)) = free_queue.pop_front() {
            if event.query() {
                if let Some(block) = blocks.get_mut(&ptr) {
                    block.is_idle = true;
                    println!("🚀 [StreamAwareAllocator] delayed_free: Block 0x{:x} is now idle (event completed)", ptr);
                }
            } else {
                remaining.push_back((event, ptr));
            }
        }
        *free_queue = remaining;
    }
}

#[pymethods]
impl StreamAwareAllocator {
    #[new]
    pub fn new() -> Self {
        Self {
            blocks: Mutex::new(HashMap::new()),
            event_queue: Mutex::new(VecDeque::new()),
            free_queue: Mutex::new(VecDeque::new()),
            next_ptr: Mutex::new(1000000),
            next_event_id: Mutex::new(1),
            stream_head: AtomicUsize::new(0),
            _pad: [0u8; 128],
            stream_tail: AtomicUsize::new(0),
            condvar: Condvar::new(),
            condvar_mutex: Mutex::new(()),
        }
    }

    pub fn process_delayed_frees(&self) -> PyResult<()> {
        let mut blocks = self.blocks.lock().unwrap();
        self.process_delayed_frees_internal(&mut blocks);
        Ok(())
    }

    pub fn allocate(&self, size: usize, stream_id: u32, tag: String) -> PyResult<usize> {
        let mut blocks = self.blocks.lock().unwrap();
        self.process_delayed_frees_internal(&mut blocks);
        
        // Stream-Aware Allocation: only reuse blocks where ALL recorded streams have completed.
        for block in blocks.values_mut() {
            if block.is_idle && block.size >= size {
                // A block is safe to reuse ONLY if:
                // 1. It has no recorded cross-stream dependencies, OR
                // 2. ALL recorded stream events have completed, OR
                // 3. The requesting stream is the same as the block's owning stream
                let safe_to_reuse = if block.recorded_streams.is_empty() {
                    true
                } else if block.recorded_streams.contains(&stream_id) && block.recorded_streams.len() == 1 {
                    // Same stream — safe (stream ordering guarantees completion)
                    true
                } else {
                    // Cross-stream reuse: check that all recorded stream events have been processed
                    // via the delayed-free queue (events marked complete)
                    let free_queue = self.free_queue.lock().unwrap();
                    let all_complete = block.recorded_streams.iter().all(|&sid| {
                        // If there are no pending events for this stream in the free queue,
                        // it means all prior events on that stream have completed
                        !free_queue.iter().any(|(evt, _)| evt.stream_id == sid && !evt.is_complete)
                    });
                    drop(free_queue);
                    all_complete
                };
                
                if safe_to_reuse {
                    println!("🚀 [StreamAwareAllocator] blockFree: Reusing block address 0x{:x} from stream {} (now stream {})", block.ptr, block.stream_id, stream_id);
                    block.is_idle = false;
                    block.stream_id = stream_id;
                    block.tag = tag.clone();
                    block.recorded_streams.clear();
                    return Ok(block.ptr);
                }
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
        let mut next_event_id = self.next_event_id.lock().unwrap();
        
        let event_id = *next_event_id;
        *next_event_id += 1;
        
        // On CPU: events complete synchronously (no async GPU stream).
        // On real GPU hardware: is_complete would start as `false` and be
        // set to `true` only when cudaEventQuery/hipEventQuery confirms
        // the stream has finished processing all prior work.
        let event = StreamEvent {
            stream_id,
            event_id,
            is_complete: true, // CPU-mode: synchronous completion
        };
        
        let mut free_queue = self.free_queue.lock().unwrap();
        free_queue.push_back((event, ptr));
        
        println!("🚀 [StreamAwareAllocator] free: Queued pointer address 0x{:x} for delayed deletion on stream {}", ptr, stream_id);
        
        Ok(())
    }

    pub fn record_stream(&self, ptr: usize, stream_id: u32) -> PyResult<()> {
        let mut blocks = self.blocks.lock().unwrap();
        if let Some(block) = blocks.get_mut(&ptr) {
            if !block.recorded_streams.contains(&stream_id) {
                block.recorded_streams.push(stream_id);
                println!("🚀 [StreamAwareAllocator] record_stream: Block 0x{:x} now tracked on stream {}", ptr, stream_id);
            }
            Ok(())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("record_stream: Block 0x{:x} not found in allocator. Cannot record stream dependency.", ptr)
            ))
        }
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

    pub fn wait_event(&self, _comm_stream_id: u32, _event: StreamEvent) -> PyResult<()> {
        // Shifting synchronization burden from CPU to GPU streams.
        // No CPU-side polling or waiting.
        Ok(())
    }

    pub fn increment_stream_head(&self) -> PyResult<()> {
        self.stream_head.fetch_add(1, Ordering::Release);
        self.condvar.notify_all();
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
        let mut guard = self.condvar_mutex.lock().unwrap();
        while self.stream_head.load(Ordering::Acquire) < target {
            guard = self.condvar.wait(guard).unwrap();
        }
        Ok(())
    }
}
