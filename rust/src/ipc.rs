use std::sync::atomic::{AtomicUsize, Ordering};
use pyo3::prelude::*;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[pyclass]
pub struct TaskMetadata {
    #[pyo3(get, set)]
    pub op_code: u32,
    #[pyo3(get, set)]
    pub tensor_id: u64,
    #[pyo3(get, set)]
    pub data_buffer: [u8; 4096],
    #[pyo3(get, set)]
    pub metadata_flat: [f32; 128],
    #[pyo3(get, set)]
    pub padding: [u8; 128],
}

impl Default for TaskMetadata {
    fn default() -> Self {
        Self {
            op_code: 0,
            tensor_id: 0,
            data_buffer: [0; 4096],
            metadata_flat: [0.0; 128],
            padding: [0; 128],
        }
    }
}

#[pymethods]
impl TaskMetadata {
    #[new]
    pub fn new() -> Self {
        Self::default()
    }

    #[getter]
    pub fn device_id(&self) -> u32 {
        self.tensor_id as u32
    }

    #[setter]
    pub fn set_device_id(&mut self, val: u32) {
        self.tensor_id = val as u64;
    }

    #[getter]
    pub fn payload(&self) -> Vec<u8> {
        self.data_buffer[..256].to_vec()
    }

    #[setter]
    pub fn set_payload(&mut self, val: Vec<u8>) {
        let copy_len = val.len().min(256);
        self.data_buffer[..copy_len].copy_from_slice(&val[..copy_len]);
        for i in copy_len..256 {
            self.data_buffer[i] = 0;
        }
    }
}

// Enforce physical separation of exactly 128 bytes of padding between atomic indices
#[repr(C)]
pub struct SPSCRingBufferLayout {
    pub head: AtomicUsize,
    pub padding: [u8; 128],
    pub tail: AtomicUsize,
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
            let fd = libc::open(path_cstr.as_ptr(), libc::O_CREAT | libc::O_RDWR | libc::O_TRUNC, 0o666);
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

        // Fallback to heap allocation using std::alloc to avoid stack overflow
        unsafe {
            let layout = std::alloc::Layout::new::<SPSCRingBufferLayout>();
            let raw_ptr = std::alloc::alloc_zeroed(layout) as *mut SPSCRingBufferLayout;
            if raw_ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            Self {
                raw_ptr,
                is_owner: true,
                is_mmap: false,
            }
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

    pub fn verify_mmap_accessibility(&self) -> PyResult<bool> {
        if !self.is_mmap || self.raw_ptr.is_null() {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Validation Failed: Shared memory segment is not actively instantiated and mapped!"
            ));
        }
        
        unsafe {
            // Verify readability and writability of head and tail atomic fields
            let layout = &*self.raw_ptr;
            let head = layout.head.load(Ordering::Acquire);
            layout.head.store(head, Ordering::Release);
            
            // Check accessibility of the buffer pages using mincore (Linux only) or a dry read/write
            #[cfg(target_os = "linux")]
            {
                let size = std::mem::size_of::<SPSCRingBufferLayout>();
                let page_size = 4096;
                let pages = (size + page_size - 1) / page_size;
                let mut vec = vec![0u8; pages];
                let res = libc::mincore(
                    self.raw_ptr as *mut libc::c_void,
                    size,
                    vec.as_mut_ptr(),
                );
                if res != 0 {
                    return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                        "Validation Failed: Shared memory segment page is not resident/accessible!"
                    ));
                }
            }
        }
        Ok(true)
    }

    pub fn verify_128_padding(&self) -> PyResult<bool> {
        unsafe {
            let layout = &*self.raw_ptr;
            let head_addr = &layout.head as *const AtomicUsize as usize;
            let tail_addr = &layout.tail as *const AtomicUsize as usize;
            let diff = tail_addr - head_addr;
            // 8 bytes for AtomicUsize + 128 bytes padding = 136 bytes difference
            Ok(diff == 136)
        }
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

impl SPSCRingBuffer {
    pub fn push(&self, op_code: u32, device_id: u32, payload_bytes: Vec<u8>) -> PyResult<()> {
        if !self.is_mmap {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "CRITICAL_FFI_SERIALIZATION_ERROR: PyBuffer is not explicitly detected as the transport layer. Pipeline aborted."
            ));
        }
        let layout = unsafe { &mut *self.raw_ptr };
        let head = layout.head.load(Ordering::Relaxed);
        
        let start = std::time::Instant::now();
        loop {
            let tail = layout.tail.load(Ordering::Acquire);
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
                        &layout.tail as *const AtomicUsize as *mut i32,
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
        let mut data_buffer = [0u8; 4096];
        let copy_len = payload_bytes.len().min(4096);
        data_buffer[..copy_len].copy_from_slice(&payload_bytes[..copy_len]);
        
        layout.buffer[index] = TaskMetadata {
            op_code,
            tensor_id: device_id as u64,
            data_buffer,
            metadata_flat: [0.0; 128],
            padding: [0; 128],
        };
        
        layout.head.store(head.wrapping_add(1), Ordering::Release);
        
        #[cfg(target_os = "linux")]
        unsafe {
            libc::syscall(
                libc::SYS_futex,
                &layout.head as *const AtomicUsize as *mut i32,
                libc::FUTEX_WAKE | libc::FUTEX_PRIVATE_FLAG,
                1 as libc::c_int,
            );
        }
        
        Ok(())
    }

    pub fn pop(&self) -> PyResult<Option<TaskMetadata>> {
        if !self.is_mmap {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "CRITICAL_FFI_SERIALIZATION_ERROR: PyBuffer is not explicitly detected as the transport layer. Pipeline aborted."
            ));
        }
        let layout = unsafe { &mut *self.raw_ptr };
        let head = layout.head.load(Ordering::Acquire);
        let tail = layout.tail.load(Ordering::Relaxed);
        
        if tail == head {
            return Ok(None);
        }
        
        let index = tail % 1024;
        let task = layout.buffer[index];
        
        layout.tail.store(tail.wrapping_add(1), Ordering::Release);
        
        #[cfg(target_os = "linux")]
        unsafe {
            libc::syscall(
                libc::SYS_futex,
                &layout.tail as *const AtomicUsize as *mut i32,
                libc::FUTEX_WAKE | libc::FUTEX_PRIVATE_FLAG,
                1 as libc::c_int,
            );
        }
        
        Ok(Some(task))
    }

    pub fn wait_and_pop(&self, py: Python<'_>) -> PyResult<TaskMetadata> {
        if !self.is_mmap {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "CRITICAL_FFI_SERIALIZATION_ERROR: PyBuffer is not explicitly detected as the transport layer. Pipeline aborted."
            ));
        }
        let layout = unsafe { &mut *self.raw_ptr };
        let tail = layout.tail.load(Ordering::Relaxed);
        
        py.allow_threads(|| {
            let start = std::time::Instant::now();
            loop {
                let current_head = layout.head.load(Ordering::Acquire);
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
                            &layout.head as *const AtomicUsize as *mut i32,
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
        layout.tail.store(tail.wrapping_add(1), Ordering::Release);
        
        #[cfg(target_os = "linux")]
        unsafe {
            libc::syscall(
                libc::SYS_futex,
                &layout.tail as *const AtomicUsize as *mut i32,
                libc::FUTEX_WAKE | libc::FUTEX_PRIVATE_FLAG,
                1 as libc::c_int,
            );
        }
        
        Ok(task)
    }
}

impl Drop for SPSCRingBuffer {
    fn drop(&mut self) {
        if self.is_owner && !self.raw_ptr.is_null() {
            unsafe {
                if self.is_mmap {
                    libc::munmap(self.raw_ptr as *mut libc::c_void, std::mem::size_of::<SPSCRingBufferLayout>());
                } else {
                    let layout = std::alloc::Layout::new::<SPSCRingBufferLayout>();
                    std::alloc::dealloc(self.raw_ptr as *mut u8, layout);
                }
            }
        }
    }
}
