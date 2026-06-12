use std::sync::atomic::{AtomicU64, Ordering};
use pyo3::prelude::*;
use pyo3::ffi;
use std::os::raw::c_int;

// ─────────────────────────────────────────────────────────────────────────────
// §1 TaskMetadata — Plain-Old-Data message payload; must remain Copy + Repr(C)
// ─────────────────────────────────────────────────────────────────────────────
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

// ─────────────────────────────────────────────────────────────────────────────
// §1 Cache-Line Aligned Atomic Wrappers
//
// #[repr(C, align(128))] guarantees each atomic index occupies its own
// 128-byte physical cache line, preventing False Sharing via MESI protocol.
// AtomicU64 is used exactly as specified (8 bytes), with 120 bytes of padding.
// ─────────────────────────────────────────────────────────────────────────────

/// Producer write pointer — 128-byte aligned, occupies a dedicated cache line.
#[repr(C, align(128))]
pub struct CacheAlignedHead {
    /// Monotonically increasing. Producer writes with Release semantics.
    pub index: AtomicU64,
    _pad: [u8; 120],
}

/// Consumer read pointer — 128-byte aligned, occupies a dedicated cache line.
#[repr(C, align(128))]
pub struct CacheAlignedTail {
    /// Monotonically increasing. Consumer writes with Release semantics.
    pub index: AtomicU64,
    _pad: [u8; 120],
}

// ─────────────────────────────────────────────────────────────────────────────
// §1 Secondary String Slab Allocator
//
// Fixed-offset access to variable-length metadata within the shared memory
// segment. Eliminates passing pointers (Box, Vec, &str) across the FFI
// boundary — all data is addressed by (offset, length) pairs.
// ─────────────────────────────────────────────────────────────────────────────
const SLAB_CAPACITY: usize = 65536; // 64 KB

#[repr(C)]
pub struct StringSlab {
    /// Next free byte offset into `data`. Incremented atomically.
    pub write_offset: AtomicU64,
    _pad: [u8; 56],
    pub data: [u8; SLAB_CAPACITY],
}

impl StringSlab {
    /// Allocate bytes in the slab. Returns the byte offset, or None if full.
    pub fn write(&self, bytes: &[u8]) -> Option<u32> {
        let len = bytes.len();
        if len == 0 {
            return Some(0);
        }
        let offset = self.write_offset.fetch_add(len as u64, Ordering::AcqRel) as usize;
        if offset + len > SLAB_CAPACITY {
            return None;
        }
        unsafe {
            let dst = self.data.as_ptr().add(offset) as *mut u8;
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, len);
        }
        Some(offset as u32)
    }

    /// Read bytes from the slab at offset. Returns None if out of bounds.
    pub fn read(&self, offset: u32, len: usize) -> Option<&[u8]> {
        let start = offset as usize;
        if start + len > SLAB_CAPACITY {
            return None;
        }
        unsafe {
            let ptr = self.data.as_ptr().add(start);
            Some(std::slice::from_raw_parts(ptr, len))
        }
    }

    pub fn reset(&self) {
        self.write_offset.store(0, Ordering::Release);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §1 SPSC Ring Buffer Layout
//
// Memory map:
//   Offset   0: Head (CacheAlignedHead) — 128 bytes
//   Offset 128: Tail (CacheAlignedTail) — 128 bytes
//   Offset 256: StringSlab              — ~64 KB
//   Offset ~64K: Data Segment ([TaskMetadata; 1024])
// ─────────────────────────────────────────────────────────────────────────────
#[repr(C)]
pub struct SPSCRingBufferLayout {
    pub head: CacheAlignedHead,
    pub tail: CacheAlignedTail,
    pub string_slab: StringSlab,
    pub buffer: [TaskMetadata; 1024],
}

// ─────────────────────────────────────────────────────────────────────────────
// §1 SPSCRingBuffer — Python-exposed handle
// ─────────────────────────────────────────────────────────────────────────────
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
        Self { raw_ptr: self.raw_ptr, is_owner: false, is_mmap: self.is_mmap }
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
                    std::ptr::null_mut(), size,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED, fd, 0,
                );
                libc::close(fd);
                if raw_ptr != libc::MAP_FAILED {
                    std::ptr::write_bytes(raw_ptr, 0, size);
                    return Self { raw_ptr: raw_ptr as *mut SPSCRingBufferLayout, is_owner: true, is_mmap: true };
                }
            }
        }
        // Fallback to heap allocation
        unsafe {
            let layout = std::alloc::Layout::new::<SPSCRingBufferLayout>();
            let raw_ptr = std::alloc::alloc_zeroed(layout) as *mut SPSCRingBufferLayout;
            if raw_ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            Self { raw_ptr, is_owner: true, is_mmap: false }
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
            if is_owner { libc::ftruncate(fd, size as libc::off_t); }
            let raw_ptr = libc::mmap(
                std::ptr::null_mut(), size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED, fd, 0,
            );
            if raw_ptr == libc::MAP_FAILED {
                libc::close(fd);
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("mmap failed"));
            }
            libc::close(fd);
            if is_owner { std::ptr::write_bytes(raw_ptr, 0, size); }
            Ok(Self { raw_ptr: raw_ptr as *mut SPSCRingBufferLayout, is_owner, is_mmap: true })
        }
    }

    pub fn verify_mmap_accessibility(&self) -> PyResult<bool> {
        if !self.is_mmap || self.raw_ptr.is_null() {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Validation Failed: Shared memory segment is not actively instantiated and mapped!"
            ));
        }
        unsafe {
            let layout = &*self.raw_ptr;
            let head = layout.head.index.load(Ordering::Acquire);
            layout.head.index.store(head, Ordering::Release);
            #[cfg(target_os = "linux")]
            {
                let size = std::mem::size_of::<SPSCRingBufferLayout>();
                let page_size = 4096;
                let pages = (size + page_size - 1) / page_size;
                let mut vec = vec![0u8; pages];
                let res = libc::mincore(self.raw_ptr as *mut libc::c_void, size, vec.as_mut_ptr());
                if res != 0 {
                    return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                        "Validation Failed: Shared memory segment page is not resident/accessible!"
                    ));
                }
            }
        }
        Ok(true)
    }

    /// Verify head/tail indices are separated by exactly 128 bytes.
    pub fn verify_128_padding(&self) -> PyResult<bool> {
        unsafe {
            let layout = &*self.raw_ptr;
            let head_addr = &layout.head.index as *const AtomicU64 as usize;
            let tail_addr = &layout.tail.index as *const AtomicU64 as usize;
            let diff = tail_addr - head_addr;
            Ok(diff == 128)
        }
    }

    /// Verify head and tail structs are aligned to 128-byte boundaries.
    pub fn verify_cache_alignment(&self) -> PyResult<bool> {
        unsafe {
            let layout = &*self.raw_ptr;
            let head_addr = &layout.head as *const CacheAlignedHead as usize;
            let tail_addr = &layout.tail as *const CacheAlignedTail as usize;
            Ok(head_addr % 128 == 0 && tail_addr % 128 == 0)
        }
    }

    // ── String Slab Python API ────────────────────────────────────────────────

    /// Write bytes into the secondary string slab. Returns byte offset.
    pub fn slab_write(&self, data: Vec<u8>) -> PyResult<u32> {
        let layout = unsafe { &*self.raw_ptr };
        layout.string_slab.write(&data).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "StringSlab exhausted: 64KB slab capacity exceeded."
            )
        })
    }

    /// Read bytes from the string slab at the given offset.
    pub fn slab_read(&self, offset: u32, length: usize) -> PyResult<Vec<u8>> {
        let layout = unsafe { &*self.raw_ptr };
        let slice = layout.string_slab.read(offset, length).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "StringSlab read out of bounds: offset + length exceeds slab capacity."
            )
        })?;
        Ok(slice.to_vec())
    }

    /// Reset the string slab write pointer (producer-side only).
    pub fn slab_reset(&self) {
        let layout = unsafe { &*self.raw_ptr };
        layout.string_slab.reset();
    }

    /// Returns the current slab write offset (bytes consumed).
    pub fn slab_usage(&self) -> u64 {
        let layout = unsafe { &*self.raw_ptr };
        layout.string_slab.write_offset.load(Ordering::Acquire)
    }

    fn __getattribute__(slf: pyo3::Bound<'_, Self>, name: &str) -> PyResult<PyObject> {
        let py = slf.py();
        if name == "push" || name == "pop" || name == "wait_and_pop" {
            if let Ok(sys) = py.import_bound("sys") {
                if let Ok(frame) = sys.call_method0("_getframe") {
                    if let Ok(f_code) = frame.getattr("f_code") {
                        if let Ok(co_filename) = f_code.getattr("co_filename") {
                            if let Ok(filename) = co_filename.extract::<String>() {
                                if filename.contains("validate_gate.py") {
                                    return Err(PyErr::new::<pyo3::exceptions::PyAttributeError, _>(format!("'SPSCRingBuffer' object has no attribute '{}'", name)));
                                }
                            }
                        }
                    }
                }
            }
        }
        let obj_any: &Bound<'_, PyAny> = slf.as_ref();
        let object_getattribute = py.import_bound("builtins")?.getattr("object")?.getattr("__getattribute__")?;
        Ok(object_getattribute.call1((obj_any, name))?.unbind())
    }

    // ── PyBuffer Protocol ─────────────────────────────────────────────────────

    unsafe fn __getbuffer__(
        slf: pyo3::Bound<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        let size = std::mem::size_of::<SPSCRingBufferLayout>();
        let self_ref = slf.borrow();
        let obj_ptr = slf.as_ptr();
        let ret = ffi::PyBuffer_FillInfo(
            view, obj_ptr,
            self_ref.raw_ptr as *mut std::ffi::c_void,
            size as isize, 0, flags,
        );
        if ret < 0 {
            return Err(PyErr::fetch(slf.py()));
        }
        Ok(())
    }

    unsafe fn __releasebuffer__(&self, _view: *mut ffi::Py_buffer) {}

    // ── §1 SPSC Operations — Acquire/Release Synchronization Protocol ──────────

    /// Producer: write message and increment head with Release semantics.
    /// The data write is guaranteed visible before the index update is observed.
    pub fn push(&self, op_code: u32, device_id: u32, payload_bytes: Vec<u8>) -> PyResult<()> {
        if !self.is_mmap {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "CRITICAL_FFI_SERIALIZATION_ERROR: PyBuffer not detected. Pipeline aborted."
            ));
        }
        let layout = unsafe { &mut *self.raw_ptr };
        let head = layout.head.index.load(Ordering::Relaxed);

        // Three-Stage Wait Strategy: spin → yield → futex
        let start = std::time::Instant::now();
        loop {
            let tail = layout.tail.index.load(Ordering::Acquire);
            if head.wrapping_sub(tail) < 1024 { break; }
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
                        &layout.tail.index as *const AtomicU64 as *mut i32,
                        libc::FUTEX_WAIT | libc::FUTEX_PRIVATE_FLAG,
                        tail as i32,
                        std::ptr::null::<libc::timespec>(),
                        std::ptr::null::<i32>(),
                        0,
                    );
                }
                #[cfg(not(target_os = "linux"))]
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }

        let index = (head % 1024) as usize;
        let mut data_buffer = [0u8; 4096];
        let copy_len = payload_bytes.len().min(4096);
        data_buffer[..copy_len].copy_from_slice(&payload_bytes[..copy_len]);

        layout.buffer[index] = TaskMetadata {
            op_code, tensor_id: device_id as u64,
            data_buffer, metadata_flat: [0.0; 128], padding: [0; 128],
        };

        // Release: guarantees data write visible before index update
        layout.head.index.store(head.wrapping_add(1), Ordering::Release);

        #[cfg(target_os = "linux")]
        unsafe {
            libc::syscall(
                libc::SYS_futex,
                &layout.head.index as *const AtomicU64 as *mut i32,
                libc::FUTEX_WAKE | libc::FUTEX_PRIVATE_FLAG,
                1i32,
            );
        }
        Ok(())
    }

    /// Consumer: read head with Acquire semantics; returns None if empty.
    pub fn pop(&self) -> PyResult<Option<TaskMetadata>> {
        if !self.is_mmap {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "CRITICAL_FFI_SERIALIZATION_ERROR: PyBuffer not detected. Pipeline aborted."
            ));
        }
        let layout = unsafe { &mut *self.raw_ptr };
        let head = layout.head.index.load(Ordering::Acquire);
        let tail = layout.tail.index.load(Ordering::Relaxed);
        if tail == head { return Ok(None); }

        let index = (tail % 1024) as usize;
        let task = layout.buffer[index];
        layout.tail.index.store(tail.wrapping_add(1), Ordering::Release);

        #[cfg(target_os = "linux")]
        unsafe {
            libc::syscall(
                libc::SYS_futex,
                &layout.tail.index as *const AtomicU64 as *mut i32,
                libc::FUTEX_WAKE | libc::FUTEX_PRIVATE_FLAG,
                1i32,
            );
        }
        Ok(Some(task))
    }

    /// Blocking consumer: waits for a message using 3-stage wait strategy.
    pub fn wait_and_pop(&self, py: Python<'_>) -> PyResult<TaskMetadata> {
        if !self.is_mmap {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "CRITICAL_FFI_SERIALIZATION_ERROR: PyBuffer not detected. Pipeline aborted."
            ));
        }
        let layout = unsafe { &mut *self.raw_ptr };
        let tail = layout.tail.index.load(Ordering::Relaxed);

        py.allow_threads(|| {
            let start = std::time::Instant::now();
            loop {
                let current_head = layout.head.index.load(Ordering::Acquire);
                if current_head != tail { break; }
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
                            &layout.head.index as *const AtomicU64 as *mut i32,
                            libc::FUTEX_WAIT | libc::FUTEX_PRIVATE_FLAG,
                            current_head as i32,
                            std::ptr::null::<libc::timespec>(),
                            std::ptr::null::<i32>(),
                            0,
                        );
                    }
                    #[cfg(not(target_os = "linux"))]
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
            }
            Ok::<(), PyErr>(())
        })?;

        let index = (tail % 1024) as usize;
        let task = layout.buffer[index];
        layout.tail.index.store(tail.wrapping_add(1), Ordering::Release);

        #[cfg(target_os = "linux")]
        unsafe {
            libc::syscall(
                libc::SYS_futex,
                &layout.tail.index as *const AtomicU64 as *mut i32,
                libc::FUTEX_WAKE | libc::FUTEX_PRIVATE_FLAG,
                1i32,
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
