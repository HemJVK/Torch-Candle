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
    is_mmap: bool,
}

unsafe impl Send for SPSCRingBuffer {}
unsafe impl Sync for SPSCRingBuffer {}

impl Clone for SPSCRingBuffer {
    fn clone(&self) -> Self {
        Self {
            raw_ptr: self.raw_ptr,
            is_owner: false,
            is_mmap: self.is_mmap,
        }
    }
}

#[pymethods]
impl SPSCRingBuffer {
    #[new]
    pub fn new() -> Self {
        let path = "/dev/shm/torch_candle_ipc";
        unsafe {
            let path_cstr = std::ffi::CString::new(path).unwrap();
            let fd = libc::open(path_cstr.as_ptr(), libc::O_CREAT | libc::O_RDWR, 0o666);
            if fd >= 0 {
                let size = std::mem::size_of::<SPSCRingBufferLayout>();
                libc::ftruncate(fd, size as libc::off_t);
                let raw_ptr = libc::mmap(
                    std::ptr::null_mut(),
                    size,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED,
                    fd,
                    0,
                );
                libc::close(fd);
                if raw_ptr != libc::MAP_FAILED {
                    std::ptr::write_bytes(raw_ptr, 0, size);
                    return Self {
                        raw_ptr: raw_ptr as *mut SPSCRingBufferLayout,
                        is_owner: true,
                        is_mmap: true,
                    };
                }
            }
        }

        // Fallback to heap allocation
        let layout = Box::new(SPSCRingBufferLayout {
            head: CacheAlignedAtomicUsize { val: AtomicUsize::new(0) },
            tail: CacheAlignedAtomicUsize { val: AtomicUsize::new(0) },
            buffer: [TaskMetadata::default(); 1024],
        });
        let raw_ptr = Box::into_raw(layout);
        Self {
            raw_ptr,
            is_owner: true,
            is_mmap: false,
        }
    }

    #[staticmethod]
    pub fn from_mmap(path: &str, is_owner: bool) -> PyResult<Self> {
        unsafe {
            let path_cstr = std::ffi::CString::new(path).unwrap();
            let fd = if is_owner {
                libc::open(path_cstr.as_ptr(), libc::O_CREAT | libc::O_RDWR | libc::O_TRUNC, 0o666)
            } else {
                libc::open(path_cstr.as_ptr(), libc::O_RDWR, 0o666)
            };
            if fd < 0 {
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to open mmap file: {}", path)));
            }
            let size = std::mem::size_of::<SPSCRingBufferLayout>();
            if is_owner {
                libc::ftruncate(fd, size as libc::off_t);
            }
            let raw_ptr = libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            );
            if raw_ptr == libc::MAP_FAILED {
                libc::close(fd);
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("mmap failed"));
            }
            libc::close(fd);
            
            if is_owner {
                std::ptr::write_bytes(raw_ptr, 0, size);
            }
            
            Ok(Self {
                raw_ptr: raw_ptr as *mut SPSCRingBufferLayout,
                is_owner,
                is_mmap: true,
            })
        }
    }

    pub fn push(&self, op_code: u32, device_id: u32, payload_bytes: Vec<u8>) -> PyResult<()> {
        let layout = unsafe { &mut *self.raw_ptr };
        let head = layout.head.val.load(Ordering::Relaxed);
        
        let start = std::time::Instant::now();
        loop {
            let tail = layout.tail.val.load(Ordering::Acquire);
            if head.wrapping_sub(tail) < 1024 {
                break;
            }
            
            let elapsed = start.elapsed().as_micros();
            if elapsed < 50 {
                std::hint::spin_loop();
            } else if elapsed < 1000 {
                std::thread::yield_now();
            } else {
                #[cfg(target_os = "linux")]
                unsafe {
                    libc::syscall(
                        libc::SYS_futex,
                        &layout.tail.val as *const AtomicUsize as *mut i32,
                        libc::FUTEX_WAIT | libc::FUTEX_PRIVATE_FLAG,
                        tail as i32,
                        std::ptr::null::<libc::timespec>(),
                        std::ptr::null::<i32>(),
                        0,
                    );
                }
                #[cfg(not(target_os = "linux"))]
                {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
            }
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
        
        #[cfg(target_os = "linux")]
        unsafe {
            libc::syscall(
                libc::SYS_futex,
                &layout.head.val as *const AtomicUsize as *mut i32,
                libc::FUTEX_WAKE | libc::FUTEX_PRIVATE_FLAG,
                1 as libc::c_int,
            );
        }
        
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
        
        #[cfg(target_os = "linux")]
        unsafe {
            libc::syscall(
                libc::SYS_futex,
                &layout.tail.val as *const AtomicUsize as *mut i32,
                libc::FUTEX_WAKE | libc::FUTEX_PRIVATE_FLAG,
                1 as libc::c_int,
            );
        }
        
        Ok(Some(task))
    }

    pub fn wait_and_pop(&self, py: Python<'_>) -> PyResult<TaskMetadata> {
        let layout = unsafe { &mut *self.raw_ptr };
        let tail = layout.tail.val.load(Ordering::Relaxed);
        
        py.allow_threads(|| {
            let start = std::time::Instant::now();
            loop {
                let current_head = layout.head.val.load(Ordering::Acquire);
                if current_head != tail {
                    break;
                }
                
                let elapsed = start.elapsed().as_micros();
                if elapsed < 50 {
                    std::hint::spin_loop();
                } else if elapsed < 1000 {
                    std::thread::yield_now();
                } else {
                    #[cfg(target_os = "linux")]
                    unsafe {
                        libc::syscall(
                            libc::SYS_futex,
                            &layout.head.val as *const AtomicUsize as *mut i32,
                            libc::FUTEX_WAIT | libc::FUTEX_PRIVATE_FLAG,
                            current_head as i32,
                            std::ptr::null::<libc::timespec>(),
                            std::ptr::null::<i32>(),
                            0,
                        );
                    }
                    #[cfg(not(target_os = "linux"))]
                    {
                        std::thread::sleep(std::time::Duration::from_millis(1));
                    }
                }
            }
            Ok::<(), PyErr>(())
        })?;
        
        let index = tail % 1024;
        let task = layout.buffer[index];
        layout.tail.val.store(tail.wrapping_add(1), Ordering::Release);
        
        #[cfg(target_os = "linux")]
        unsafe {
            libc::syscall(
                libc::SYS_futex,
                &layout.tail.val as *const AtomicUsize as *mut i32,
                libc::FUTEX_WAKE | libc::FUTEX_PRIVATE_FLAG,
                1 as libc::c_int,
            );
        }
        
        Ok(task)
    }

    unsafe fn __getbuffer__(
        &self,
        view: *mut pyo3::ffi::Py_buffer,
        flags: std::os::raw::c_int,
    ) -> PyResult<()> {
        let size = std::mem::size_of::<SPSCRingBufferLayout>();
        pyo3::ffi::PyBuffer_FillInfo(
            view,
            std::ptr::null_mut(),
            self.raw_ptr as *mut std::ffi::c_void,
            size as isize,
            0,
            flags,
        );
        Ok(())
    }

    unsafe fn __releasebuffer__(&self, _view: *mut pyo3::ffi::Py_buffer) {}
}

impl Drop for SPSCRingBuffer {
    fn drop(&mut self) {
        if self.is_owner {
            unsafe {
                if self.is_mmap {
                    libc::munmap(self.raw_ptr as *mut libc::c_void, std::mem::size_of::<SPSCRingBufferLayout>());
                } else {
                    let _ = Box::from_raw(self.raw_ptr);
                }
            }
        }
    }
}
