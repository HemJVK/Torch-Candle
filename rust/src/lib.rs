use numpy::{PyReadonlyArrayDyn, ToPyArray, PyUntypedArrayMethods};
use pyo3::prelude::*;
use candle_core::{Tensor, Device, DType};
use std::sync::Arc;
use parking_lot::Mutex;
mod kernels;
mod simd;

// --- Autograd Infrastructure ---

pub trait OpNode: Send + Sync {
    fn name(&self) -> &str;
    fn backward(&self, grad: &Tensor) -> Vec<Option<Tensor>>;
}

struct AddNode {
    lhs_req: bool,
    rhs_req: bool,
}
impl OpNode for AddNode {
    fn name(&self) -> &str { "Add" }
    fn backward(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        let mut grads = Vec::new();
        if self.lhs_req { grads.push(Some(grad.clone())); } else { grads.push(None); }
        if self.rhs_req { grads.push(Some(grad.clone())); } else { grads.push(None); }
        grads
    }
}

struct SubNode {
    lhs_req: bool,
    rhs_req: bool,
}
impl OpNode for SubNode {
    fn name(&self) -> &str { "Sub" }
    fn backward(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        let mut grads = Vec::new();
        if self.lhs_req { grads.push(Some(grad.clone())); } else { grads.push(None); }
        if self.rhs_req { grads.push(Some(grad.neg().unwrap())); } else { grads.push(None); }
        grads
    }
}

struct MulNode {
    lhs: Tensor,
    rhs: Tensor,
    lhs_req: bool,
    rhs_req: bool,
}
impl OpNode for MulNode {
    fn name(&self) -> &str { "Mul" }
    fn backward(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        let mut grads = Vec::new();
        if self.lhs_req { grads.push(Some(grad.broadcast_mul(&self.rhs).unwrap())); } else { grads.push(None); }
        if self.rhs_req { grads.push(Some(grad.broadcast_mul(&self.lhs).unwrap())); } else { grads.push(None); }
        grads
    }
}

struct DivNode {
    lhs: Tensor,
    rhs: Tensor,
    lhs_req: bool,
    rhs_req: bool,
}
impl OpNode for DivNode {
    fn name(&self) -> &str { "Div" }
    fn backward(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        let mut grads = Vec::new();
        if self.lhs_req {
            // d(x/y)/dx = 1/y
            grads.push(Some(grad.broadcast_div(&self.rhs).unwrap()));
        } else { grads.push(None); }
        if self.rhs_req {
            // d(x/y)/dy = -x / y^2
            let y2 = self.rhs.sqr().unwrap();
            let neg_x = self.lhs.neg().unwrap();
            let dy = neg_x.broadcast_div(&y2).unwrap();
            grads.push(Some(grad.broadcast_mul(&dy).unwrap()));
        } else { grads.push(None); }
        grads
    }
}

struct MatmulNode {
    lhs: Tensor,
    rhs: Tensor,
    lhs_req: bool,
    rhs_req: bool,
}
impl OpNode for MatmulNode {
    fn name(&self) -> &str { "Matmul" }
    fn backward(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        let mut grads = Vec::new();
        if self.lhs_req {
            // d(X@W)/dX = grad @ W.T
            let rhs_t = self.rhs.t().unwrap();
            grads.push(Some(grad.matmul(&rhs_t).unwrap()));
        } else { grads.push(None); }
        if self.rhs_req {
            // d(X@W)/dW = X.T @ grad
            let lhs_t = self.lhs.t().unwrap();
            grads.push(Some(lhs_t.matmul(grad).unwrap()));
        } else { grads.push(None); }
        grads
    }
}

struct SumNode {
    input_shape: Vec<usize>,
}
impl OpNode for SumNode {
    fn name(&self) -> &str { "Sum" }
    fn backward(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        // d(sum(x))/dx = 1 (broadcast to x shape)
        let ones = Tensor::ones(self.input_shape.as_slice(), grad.dtype(), grad.device()).unwrap();
        vec![Some(ones.broadcast_mul(grad).unwrap())]
    }
}

struct MeanNode {
    input_shape: Vec<usize>,
}
impl OpNode for MeanNode {
    fn name(&self) -> &str { "Mean" }
    fn backward(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        let n = self.input_shape.iter().product::<usize>() as f64;
        let factor = Tensor::new(&[1.0 / n as f32], grad.device()).unwrap().to_dtype(grad.dtype()).unwrap();
        let ones = Tensor::ones(self.input_shape.as_slice(), grad.dtype(), grad.device()).unwrap();
        let item_grad = ones.broadcast_mul(&factor).unwrap();
        vec![Some(item_grad.broadcast_mul(grad).unwrap())]
    }
}

struct ReshapeNode {
    input_shape: Vec<usize>,
}
impl OpNode for ReshapeNode {
    fn name(&self) -> &str { "Reshape" }
    fn backward(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        vec![Some(grad.reshape(self.input_shape.as_slice()).unwrap())]
    }
}

struct TransposeNode {
    dim0: usize,
    dim1: usize,
}
impl OpNode for TransposeNode {
    fn name(&self) -> &str { "Transpose" }
    fn backward(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        vec![Some(grad.transpose(self.dim0, self.dim1).unwrap())]
    }
}

struct TNode;
impl OpNode for TNode {
    fn name(&self) -> &str { "Transpose(T)" }
    fn backward(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        vec![Some(grad.t().unwrap())]
    }
}

struct ReluNode {
    input: Tensor,
}
impl OpNode for ReluNode {
    fn name(&self) -> &str { "Relu" }
    fn backward(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        // d(relu(x))/dx = 1 if x > 0 else 0
        let mask = self.input.gt(0.0).unwrap().to_dtype(self.input.dtype()).unwrap();
        vec![Some(grad.mul(&mask).unwrap())]
    }
}

struct SinNode {
    input: Tensor,
}
impl OpNode for SinNode {
    fn name(&self) -> &str { "Sin" }
    fn backward(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        // d(sin(x))/dx = cos(x)
        let cos = self.input.cos().unwrap();
        vec![Some(grad.mul(&cos).unwrap())]
    }
}

struct CosNode {
    input: Tensor,
}
impl OpNode for CosNode {
    fn name(&self) -> &str { "Cos" }
    fn backward(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        // d(cos(x))/dx = -sin(x)
        let sin = self.input.sin().unwrap();
        let neg_sin = sin.neg().unwrap();
        vec![Some(grad.mul(&neg_sin).unwrap())]
    }
}

struct RecipNode {
    input: Tensor,
}
impl OpNode for RecipNode {
    fn name(&self) -> &str { "Recip" }
    fn backward(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        let x2 = self.input.sqr().unwrap();
        let neg_one = Tensor::new(&[-1.0f32], self.input.device()).unwrap().to_dtype(self.input.dtype()).unwrap();
        let d = neg_one.broadcast_div(&x2).unwrap();
        vec![Some(grad.mul(&d).unwrap())]
    }
}

struct PowNode {
    input: Tensor,
    exponent: f64,
}
impl OpNode for PowNode {
    fn name(&self) -> &str { "Pow" }
    fn backward(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        // d(x^n)/dx = n * x^(n-1)
        let n = self.exponent as f32;
        let n_minus_1 = n - 1.0;
        let x_n_minus_1 = self.input.powf(n_minus_1 as f64).unwrap();
        let n_t = Tensor::new(&[n], grad.device()).unwrap().to_dtype(grad.dtype()).unwrap();
        let d = x_n_minus_1.broadcast_mul(&n_t).unwrap();
        vec![Some(grad.broadcast_mul(&d).unwrap())]
    }
}

struct AbsNode {
    input: Tensor,
}
impl OpNode for AbsNode {
    fn name(&self) -> &str { "Abs" }
    fn backward(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        // d(|x|)/dx = sign(x)
        let sign = self.input.broadcast_div(&self.input.abs().unwrap()).unwrap(); // Simple sign
        vec![Some(grad.broadcast_mul(&sign).unwrap())]
    }
}

struct ClampNode {
    input: Tensor,
    min: f64,
    max: f64,
}
impl OpNode for ClampNode {
    fn name(&self) -> &str { "Clamp" }
    fn backward(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        // gradient is 1 if within [min, max], else 0
        let mask_min = self.input.ge(self.min).unwrap().to_dtype(grad.dtype()).unwrap();
        let mask_max = self.input.le(self.max).unwrap().to_dtype(grad.dtype()).unwrap();
        let mask = mask_min.broadcast_mul(&mask_max).unwrap();
        vec![Some(grad.broadcast_mul(&mask).unwrap())]
    }
}
struct NarrowNode {
    dim: usize,
    start: usize,
    len: usize,
}
impl OpNode for NarrowNode {
    fn name(&self) -> &str { "Narrow" }
    fn backward(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        // gradient for narrow is a zero tensor with grad placed back at the narrow window
        // but candle's Tensor doesn't have an easy "scatter" or "index_fill" that works easily here?
        // Actually, we can use zeros and then index_add? No, padding is easier.
        // For now, let's use a zero tensor and index_add.
        vec![None] // TODO: implement full narrow backward logic
    }
}
struct CatNode {
    dim: usize,
    shapes: Vec<Vec<usize>>,
}
impl OpNode for CatNode {
    fn name(&self) -> &str { "Cat" }
    fn backward(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        let mut start = 0;
        let mut grads = Vec::new();
        for shape in &self.shapes {
            let len = shape[self.dim];
            let g = grad.narrow(self.dim, start, len).unwrap();
            grads.push(Some(g));
            start += len;
        }
        grads
    }
}

struct IndexSelectNode {
    input_shape: Vec<usize>,
    index: Tensor,
    dim: usize,
}
impl OpNode for IndexSelectNode {
    fn name(&self) -> &str { "IndexSelect" }
    fn backward(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        // gradient for index_select is scatter_add (basically putting grad back into zero tensor)
        let mut input_grad = Tensor::zeros(self.input_shape.as_slice(), grad.dtype(), grad.device()).unwrap();
        input_grad = input_grad.index_add(&self.index, grad, self.dim).unwrap();
        vec![Some(input_grad)]
    }
}

struct StackNode {
    dim: usize,
    shapes: Vec<Vec<usize>>,
}
impl OpNode for StackNode {
    fn name(&self) -> &str { "Stack" }
    fn backward(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        let mut grads = Vec::new();
        for i in 0..self.shapes.len() {
            let g = grad.narrow(self.dim, i, 1).unwrap().squeeze(self.dim).unwrap();
            grads.push(Some(g));
        }
        grads
    }
}

struct WhereNode {
    cond: Tensor,
    input_req: bool,
    other_req: bool,
}
impl OpNode for WhereNode {
    fn name(&self) -> &str { "Where" }
    fn backward(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        let mut grads = Vec::new();
        if self.input_req {
            let zero = Tensor::zeros_like(grad).unwrap();
            grads.push(Some(self.cond.where_cond(grad, &zero).unwrap()));
        } else { grads.push(None); }
        if self.other_req {
            let zero = Tensor::zeros_like(grad).unwrap();
            grads.push(Some(self.cond.where_cond(&zero, grad).unwrap()));
        } else { grads.push(None); }
        grads
    }
}
#[pyclass]
#[derive(Clone)]
pub struct PyTensor {
    inner: Tensor,
    grad: Option<Arc<Mutex<Option<Tensor>>>>,
    grad_fn: Option<Arc<dyn OpNode>>,
    #[pyo3(get, set)]
    requires_grad: bool,
    parents: Vec<PyTensor>,
}

#[pymethods]
impl PyTensor {
    #[new]
    #[pyo3(signature = (data, device="cpu", dtype="float32", requires_grad=false))]
    fn new(data: PyReadonlyArrayDyn<'_, f32>, device: &str, dtype: &str, requires_grad: bool) -> PyResult<Self> {
        let shape = data.shape();
        let slice = data.as_slice().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        
        let dev = match device {
            "cpu" => Device::Cpu,
            "cuda" => Device::new_cuda(0).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?,
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Unsupported device: {}", device))),
        };

        let dt = match dtype {
            "float32" => DType::F32,
            "float64" => DType::F64,
            "uint32" => DType::U32,
            "uint8" => DType::U8,
            "int64" => DType::I64,
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Unsupported dtype: {}", dtype))),
        };

        let inner = Tensor::from_slice(slice, shape, &dev)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?
            .to_dtype(dt)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;

        let grad = if requires_grad { Some(Arc::new(Mutex::new(None))) } else { None };

        Ok(PyTensor { 
            inner, 
            grad, 
            grad_fn: None, 
            requires_grad,
            parents: Vec::new(),
        })
    }

    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.dims().to_vec()
    }

    #[getter]
    fn device(&self) -> String {
        match self.inner.device() {
            Device::Cpu => "cpu".to_string(),
            Device::Cuda(_) => "cuda".to_string(),
            Device::Metal(_) => "metal".to_string(),
        }
    }

    fn to_numpy(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dtype = self.inner.dtype();
        if dtype != DType::F32 {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("Only F32 supported for to_numpy for now"));
        }
        let dims = self.inner.dims();
        let data = self.inner.flatten_all().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?
            .to_vec1::<f32>().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
        
        let array = ndarray::ArrayD::from_shape_vec(dims, data).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        Ok(array.to_pyarray_bound(py).into())
    }

    #[staticmethod]
    #[pyo3(signature = (shape, device="cpu", dtype="float32"))]
    fn ones(shape: Vec<usize>, device: &str, dtype: &str) -> PyResult<Self> {
        let dev = match device {
            "cpu" => Device::Cpu,
            "cuda" => Device::new_cuda(0).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?,
            _ => Device::Cpu,
        };
        let dt = match dtype {
            "float32" => DType::F32,
            "float64" => DType::F64,
            _ => DType::F32,
        };
        let inner = Tensor::ones(shape.as_slice(), dt, &dev).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
        Ok(PyTensor { inner, grad: None, grad_fn: None, requires_grad: false, parents: Vec::new() })
    }

    #[staticmethod]
    #[pyo3(signature = (shape, device="cpu", dtype="float32"))]
    fn zeros(shape: Vec<usize>, device: &str, dtype: &str) -> PyResult<Self> {
        let dev = match device {
            "cpu" => Device::Cpu,
            "cuda" => Device::new_cuda(0).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?,
            _ => Device::Cpu,
        };
        let dt = match dtype {
            "float32" => DType::F32,
            "float64" => DType::F64,
            _ => DType::F32,
        };
        let inner = Tensor::zeros(shape.as_slice(), dt, &dev).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
        Ok(PyTensor { inner, grad: None, grad_fn: None, requires_grad: false, parents: Vec::new() })
    }

    #[staticmethod]
    fn cat(tensors: Vec<PyTensor>, dim: usize) -> PyResult<Self> {
        let inners: Vec<Tensor> = tensors.iter().map(|t| t.inner.clone()).collect();
        let inner = Tensor::cat(&inners, dim).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
        
        let requires_grad = tensors.iter().any(|t| t.requires_grad);
        let mut grad_fn = None;
        let mut parents = Vec::new();

        if requires_grad {
            let shapes = tensors.iter().map(|t| t.inner.dims().to_vec()).collect();
            grad_fn = Some(Arc::new(CatNode { dim, shapes }) as Arc<dyn OpNode>);
            parents = tensors.clone();
        }

        Ok(PyTensor { 
            inner, 
            grad: if requires_grad { Some(Arc::new(Mutex::new(None))) } else { None },
            grad_fn,
            requires_grad,
            parents,
        })
    }

    #[staticmethod]
    fn stack(tensors: Vec<PyTensor>, dim: usize) -> PyResult<Self> {
        let inners: Vec<Tensor> = tensors.iter().map(|t| t.inner.clone()).collect();
        let inner = Tensor::stack(&inners, dim).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
        
        let requires_grad = tensors.iter().any(|t| t.requires_grad);
        let mut grad_fn = None;
        let mut parents = Vec::new();

        if requires_grad {
            let shapes = tensors.iter().map(|t| t.inner.dims().to_vec()).collect();
            grad_fn = Some(Arc::new(StackNode { dim, shapes }) as Arc<dyn OpNode>);
            parents = tensors.clone();
        }

        Ok(PyTensor { 
            inner, 
            grad: if requires_grad { Some(Arc::new(Mutex::new(None))) } else { None },
            grad_fn,
            requires_grad,
            parents,
        })
    }

    fn to_dtype(&self, dtype: &str) -> PyResult<Self> {
        let dt = match dtype {
            "float32" => DType::F32,
            "float64" => DType::F64,
            "uint32" => DType::U32,
            "uint8" => DType::U8,
            _ => DType::F32,
        };
        let inner = self.inner.to_dtype(dt).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
        Ok(PyTensor { inner, grad: self.grad.clone(), grad_fn: self.grad_fn.clone(), requires_grad: self.requires_grad, parents: self.parents.clone() })
    }

    fn narrow(&self, dim: usize, start: usize, len: usize) -> PyResult<Self> {
        let inner = self.inner.narrow(dim, start, len).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
        let requires_grad = self.requires_grad;
        let mut grad_fn = None;
        let mut parents = Vec::new();
        if requires_grad {
            // We need a proper Narrow backward node. For now, we'll placeholder it.
            // But if we want it to work in the test, we should at least track it.
            grad_fn = Some(Arc::new(NarrowNode { dim, start, len }) as Arc<dyn OpNode>);
            parents.push(self.clone());
        }
        Ok(PyTensor { inner, grad: if requires_grad { Some(Arc::new(Mutex::new(None))) } else { None }, grad_fn, requires_grad, parents })
    }

    fn get(&self, index: usize) -> PyResult<Self> {
        let inner = self.inner.get(index).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
        Ok(PyTensor { inner, grad: None, grad_fn: None, requires_grad: false, parents: Vec::new() })
    }

    fn where_cond(&self, input: &PyTensor, other: &PyTensor) -> PyResult<Self> {
        let inner = self.inner.where_cond(&input.inner, &other.inner).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
        
        let requires_grad = input.requires_grad || other.requires_grad;
        let mut grad_fn = None;
        let mut parents = Vec::new();

        if requires_grad {
            grad_fn = Some(Arc::new(WhereNode { 
                cond: self.inner.clone(), 
                input_req: input.requires_grad, 
                other_req: other.requires_grad 
            }) as Arc<dyn OpNode>);
            parents.push(input.clone());
            parents.push(other.clone());
        }

        Ok(PyTensor { 
            inner, 
            grad: if requires_grad { Some(Arc::new(Mutex::new(None))) } else { None },
            grad_fn,
            requires_grad,
            parents,
        })
    }

    fn index_select(&self, index: &PyTensor, dim: usize) -> PyResult<Self> {
        let inner = self.inner.index_select(&index.inner, dim).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
        
        let requires_grad = self.requires_grad;
        let mut grad_fn = None;
        let mut parents = Vec::new();

        if requires_grad {
            grad_fn = Some(Arc::new(IndexSelectNode { 
                input_shape: self.inner.dims().to_vec(), 
                index: index.inner.clone(), 
                dim 
            }) as Arc<dyn OpNode>);
            parents.push(self.clone());
        }

        Ok(PyTensor { 
            inner, 
            grad: if requires_grad { Some(Arc::new(Mutex::new(None))) } else { None },
            grad_fn,
            requires_grad,
            parents,
        })
    }

    fn argmax_keepdim(&self, dim: usize) -> PyResult<Self> {
        let inner = self.inner.argmax_keepdim(dim).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
        Ok(PyTensor { inner, grad: None, grad_fn: None, requires_grad: false, parents: Vec::new() })
    }

    fn argmin_keepdim(&self, dim: usize) -> PyResult<Self> {
        let inner = self.inner.argmin_keepdim(dim).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
        Ok(PyTensor { inner, grad: None, grad_fn: None, requires_grad: false, parents: Vec::new() })
    }

    fn max_keepdim(&self, dim: usize) -> PyResult<Self> {
        let inner = self.inner.max_keepdim(dim).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
        Ok(PyTensor { inner, grad: None, grad_fn: None, requires_grad: false, parents: Vec::new() })
    }

    fn min_keepdim(&self, dim: usize) -> PyResult<Self> {
        let inner = self.inner.min_keepdim(dim).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
        Ok(PyTensor { inner, grad: None, grad_fn: None, requires_grad: false, parents: Vec::new() })
    }

    fn flatten_all(&self) -> PyResult<Self> {
        let inner = self.inner.flatten_all().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
        let requires_grad = self.requires_grad;
        let mut grad_fn = None;
        let mut parents = Vec::new();
        if requires_grad {
            grad_fn = Some(Arc::new(ReshapeNode { input_shape: self.inner.dims().to_vec() }) as Arc<dyn OpNode>);
            parents.push(self.clone());
        }
        Ok(PyTensor { inner, grad: if requires_grad { Some(Arc::new(Mutex::new(None))) } else { None }, grad_fn, requires_grad, parents })
    }

    fn squeeze(&self, dim: usize) -> PyResult<Self> {
        let inner = self.inner.squeeze(dim).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
        let requires_grad = self.requires_grad;
        let mut grad_fn = None;
        let mut parents = Vec::new();
        if requires_grad {
            grad_fn = Some(Arc::new(ReshapeNode { input_shape: self.inner.dims().to_vec() }) as Arc<dyn OpNode>);
            parents.push(self.clone());
        }
        Ok(PyTensor { inner, grad: if requires_grad { Some(Arc::new(Mutex::new(None))) } else { None }, grad_fn, requires_grad, parents })
    }

    fn unsqueeze(&self, dim: usize) -> PyResult<Self> {
        let inner = self.inner.unsqueeze(dim).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
        let requires_grad = self.requires_grad;
        let mut grad_fn = None;
        let mut parents = Vec::new();
        if requires_grad {
            grad_fn = Some(Arc::new(ReshapeNode { input_shape: self.inner.dims().to_vec() }) as Arc<dyn OpNode>);
            parents.push(self.clone());
        }
        Ok(PyTensor { inner, grad: if requires_grad { Some(Arc::new(Mutex::new(None))) } else { None }, grad_fn, requires_grad, parents })
    }

    #[getter]
    fn grad(&self) -> PyResult<Option<PyTensor>> {
        if let Some(ref g_mutex) = self.grad {
            let g_opt = g_mutex.lock();
            if let Some(ref g) = *g_opt {
                return Ok(Some(PyTensor {
                    inner: g.clone(),
                    grad: None,
                    grad_fn: None,
                    requires_grad: false,
                    parents: Vec::new(),
                }));
            }
        }
        Ok(None)
    }

    #[setter]
    fn set_grad(&self, new_grad: Option<PyTensor>) -> PyResult<()> {
        if let Some(ref g_mutex) = self.grad {
            let mut g_opt = g_mutex.lock();
            *g_opt = new_grad.map(|t| t.inner.clone());
            Ok(())
        } else {
            if new_grad.is_some() {
                Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("cannot set grad on tensor that does not require grad"))
            } else {
                Ok(())
            }
        }
    }

    // --- Arithmetic ---
    fn add(&self, other: &PyTensor) -> PyResult<Self> {
        let (lhs_t, rhs_t) = self.broadcast_to_same_rank(&other.inner)?;
        let inner = lhs_t.broadcast_add(&rhs_t).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
        
        let requires_grad = self.requires_grad || other.requires_grad;
        let mut grad_fn = None;
        let mut parents = Vec::new();

        if requires_grad {
            grad_fn = Some(Arc::new(AddNode { lhs_req: self.requires_grad, rhs_req: other.requires_grad }) as Arc<dyn OpNode>);
            parents.push(self.clone());
            parents.push(other.clone());
        }

        Ok(PyTensor { 
            inner, 
            grad: if requires_grad { Some(Arc::new(Mutex::new(None))) } else { None },
            grad_fn,
            requires_grad,
            parents,
        })
    }

    fn sub(&self, other: &PyTensor) -> PyResult<Self> {
        let (lhs_t, rhs_t) = self.broadcast_to_same_rank(&other.inner)?;
        let inner = lhs_t.broadcast_sub(&rhs_t).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
        
        let requires_grad = self.requires_grad || other.requires_grad;
        let mut grad_fn = None;
        let mut parents = Vec::new();

        if requires_grad {
            grad_fn = Some(Arc::new(SubNode { lhs_req: self.requires_grad, rhs_req: other.requires_grad }) as Arc<dyn OpNode>);
            parents.push(self.clone());
            parents.push(other.clone());
        }

        Ok(PyTensor { 
            inner, 
            grad: if requires_grad { Some(Arc::new(Mutex::new(None))) } else { None },
            grad_fn,
            requires_grad,
            parents,
        })
    }

    fn abs(&self) -> PyResult<Self> {
        let inner = self.inner.abs().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
        let requires_grad = self.requires_grad;
        let mut grad_fn = None;
        let mut parents = Vec::new();
        if requires_grad {
            grad_fn = Some(Arc::new(AbsNode { input: self.inner.clone() }) as Arc<dyn OpNode>);
            parents.push(self.clone());
        }
        Ok(PyTensor { inner, grad: if requires_grad { Some(Arc::new(Mutex::new(None))) } else { None }, grad_fn, requires_grad, parents })
    }

    fn clamp(&self, min: f64, max: f64) -> PyResult<Self> {
        let inner = self.inner.clamp(min, max).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
        let requires_grad = self.requires_grad;
        let mut grad_fn = None;
        let mut parents = Vec::new();
        if requires_grad {
            grad_fn = Some(Arc::new(ClampNode { input: self.inner.clone(), min, max }) as Arc<dyn OpNode>);
            parents.push(self.clone());
        }
        Ok(PyTensor { inner, grad: if requires_grad { Some(Arc::new(Mutex::new(None))) } else { None }, grad_fn, requires_grad, parents })
    }
    #[pyo3(name = "pow")]
    fn pow(&self, exponent: f64) -> PyResult<Self> {
        let inner = self.inner.powf(exponent).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
        let requires_grad = self.requires_grad;
        let mut grad_fn = None;
        let mut parents = Vec::new();
        if requires_grad {
            grad_fn = Some(Arc::new(PowNode { input: self.inner.clone(), exponent }) as Arc<dyn OpNode>);
            parents.push(self.clone());
        }
        Ok(PyTensor { inner, grad: if requires_grad { Some(Arc::new(Mutex::new(None))) } else { None }, grad_fn, requires_grad, parents })
    }

    fn mul(&self, other: &PyTensor) -> PyResult<Self> {
        let (lhs_t, rhs_t) = self.broadcast_to_same_rank(&other.inner)?;
        let inner = lhs_t.broadcast_mul(&rhs_t).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
        
        let requires_grad = self.requires_grad || other.requires_grad;
        let mut grad_fn = None;
        let mut parents = Vec::new();

        if requires_grad {
            grad_fn = Some(Arc::new(MulNode { 
                lhs: self.inner.clone(), 
                rhs: other.inner.clone(),
                lhs_req: self.requires_grad, 
                rhs_req: other.requires_grad 
            }) as Arc<dyn OpNode>);
            parents.push(self.clone());
            parents.push(other.clone());
        }

        Ok(PyTensor { 
            inner, 
            grad: if requires_grad { Some(Arc::new(Mutex::new(None))) } else { None },
            grad_fn,
            requires_grad,
            parents,
        })
    }

    fn div(&self, other: &PyTensor) -> PyResult<Self> {
        let (lhs_t, rhs_t) = self.broadcast_to_same_rank(&other.inner)?;
        let inner = lhs_t.broadcast_div(&rhs_t).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
        
        let requires_grad = self.requires_grad || other.requires_grad;
        let mut grad_fn = None;
        let mut parents = Vec::new();

        if requires_grad {
            grad_fn = Some(Arc::new(DivNode { 
                lhs: self.inner.clone(), 
                rhs: other.inner.clone(),
                lhs_req: self.requires_grad, 
                rhs_req: other.requires_grad 
            }) as Arc<dyn OpNode>);
            parents.push(self.clone());
            parents.push(other.clone());
        }

        Ok(PyTensor { 
            inner, 
            grad: if requires_grad { Some(Arc::new(Mutex::new(None))) } else { None },
            grad_fn,
            requires_grad,
            parents,
        })
    }

    fn matmul(&self, other: &PyTensor) -> PyResult<Self> {
        let inner = self.inner.matmul(&other.inner).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
        
        let requires_grad = self.requires_grad || other.requires_grad;
        let mut grad_fn = None;
        let mut parents = Vec::new();

        if requires_grad {
            grad_fn = Some(Arc::new(MatmulNode { 
                lhs: self.inner.clone(), 
                rhs: other.inner.clone(),
                lhs_req: self.requires_grad, 
                rhs_req: other.requires_grad 
            }) as Arc<dyn OpNode>);
            parents.push(self.clone());
            parents.push(other.clone());
        }

        Ok(PyTensor { 
            inner, 
            grad: if requires_grad { Some(Arc::new(Mutex::new(None))) } else { None },
            grad_fn,
            requires_grad,
            parents,
        })
    }

    // --- Reductions ---
    fn sum_all(&self) -> PyResult<Self> {
        let inner = self.inner.sum_all().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
        let requires_grad = self.requires_grad;
        let mut grad_fn = None;
        let mut parents = Vec::new();

        if requires_grad {
            grad_fn = Some(Arc::new(SumNode { input_shape: self.inner.dims().to_vec() }) as Arc<dyn OpNode>);
            parents.push(self.clone());
        }

        Ok(PyTensor { 
            inner, 
            grad: if requires_grad { Some(Arc::new(Mutex::new(None))) } else { None },
            grad_fn,
            requires_grad,
            parents,
        })
    }

    fn mean_all(&self) -> PyResult<Self> {
        let inner = self.inner.mean_all().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
        let requires_grad = self.requires_grad;
        let mut grad_fn = None;
        let mut parents = Vec::new();

        if requires_grad {
            grad_fn = Some(Arc::new(MeanNode { input_shape: self.inner.dims().to_vec() }) as Arc<dyn OpNode>);
            parents.push(self.clone());
        }

        Ok(PyTensor { 
            inner, 
            grad: if requires_grad { Some(Arc::new(Mutex::new(None))) } else { None },
            grad_fn,
            requires_grad,
            parents,
        })
    }

    // ... (reshape, t, transpose, etc. should also record parents if requires_grad) ...
    // Skipping others for brevity in this step, but standard logic applies.

    #[pyo3(signature = (gradient=None))]
    fn backward(&self, _py: Python<'_>, gradient: Option<PyTensor>) -> PyResult<()> {
        if !self.requires_grad {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("backward called on tensor that does not require grad"));
        }

        let grad_val = match gradient {
            Some(g) => g.inner.clone(),
            None => Tensor::ones_like(&self.inner).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?,
        };

        // Accumulate grad on self
        if let Some(ref g_mutex) = self.grad {
            let mut g_opt = g_mutex.lock();
            if let Some(ref current_g) = *g_opt {
                *g_opt = Some(current_g.add(&grad_val).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?);
            } else {
                *g_opt = Some(grad_val.clone());
            }
        }

        // --- Topological Engine ---
        let mut visited_ptrs = std::collections::HashSet::<*const ()>::new();
        let mut topo_order = Vec::new();

        fn build_topo(tensor: &PyTensor, visited: &mut std::collections::HashSet<*const ()>, topo: &mut Vec<PyTensor>) {
            // Only add to topo if it has a grad_fn, otherwise it's a leaf or doesn't require grad
            if tensor.grad_fn.is_some() {
                let ptr = Arc::as_ptr(tensor.grad.as_ref().unwrap()) as *const (); // Using grad Arc ptr as unique ID
                if visited.contains(&ptr) { return; }
                visited.insert(ptr);
                for parent in &tensor.parents {
                    if parent.requires_grad {
                        build_topo(parent, visited, topo);
                    }
                }
                topo.push(tensor.clone());
            }
        }
        
        build_topo(self, &mut visited_ptrs, &mut topo_order);
        topo_order.reverse(); // Now it's from output to inputs

        for tensor in topo_order {
            let current_grad = {
                let g_mutex = tensor.grad.as_ref().ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Tensor in topo order lacks grad Arc"))?;
                let g_opt = g_mutex.lock();
                match *g_opt {
                    Some(ref g) => g.clone(),
                    None => continue, // No gradient reached this node yet
                }
            };

            if let Some(ref node) = tensor.grad_fn {
                let parent_grads = node.backward(&current_grad);
                for (parent, maybe_grad) in tensor.parents.iter().zip(parent_grads) {
                    if let Some(p_grad) = maybe_grad {
                        if let Some(ref pg_mutex) = parent.grad {
                            let mut pg_opt = pg_mutex.lock();
                            if let Some(ref current_pg) = *pg_opt {
                                *pg_opt = Some(current_pg.add(&p_grad).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?);
                            } else {
                                *pg_opt = Some(p_grad);
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn reshape(&self, shape: Vec<usize>) -> PyResult<Self> {
        let inner = self.inner.reshape(shape.as_slice()).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
        let requires_grad = self.requires_grad;
        let mut grad_fn = None;
        let mut parents = Vec::new();

        if requires_grad {
            grad_fn = Some(Arc::new(ReshapeNode { input_shape: self.inner.dims().to_vec() }) as Arc<dyn OpNode>);
            parents.push(self.clone());
        }

        Ok(PyTensor { 
            inner, 
            grad: if requires_grad { Some(Arc::new(Mutex::new(None))) } else { None },
            grad_fn,
            requires_grad,
            parents,
        })
    }

    fn t(&self) -> PyResult<Self> {
        let inner = self.inner.t().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
        let requires_grad = self.requires_grad;
        let mut grad_fn = None;
        let mut parents = Vec::new();

        if requires_grad {
            grad_fn = Some(Arc::new(TNode) as Arc<dyn OpNode>);
            parents.push(self.clone());
        }

        Ok(PyTensor { 
            inner, 
            grad: if requires_grad { Some(Arc::new(Mutex::new(None))) } else { None },
            grad_fn,
            requires_grad,
            parents,
        })
    }

    fn transpose(&self, dim0: usize, dim1: usize) -> PyResult<Self> {
        let inner = self.inner.transpose(dim0, dim1).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
        let requires_grad = self.requires_grad;
        let mut grad_fn = None;
        let mut parents = Vec::new();

        if requires_grad {
            grad_fn = Some(Arc::new(TransposeNode { dim0, dim1 }) as Arc<dyn OpNode>);
            parents.push(self.clone());
        }

        Ok(PyTensor { 
            inner, 
            grad: if requires_grad { Some(Arc::new(Mutex::new(None))) } else { None },
            grad_fn,
            requires_grad,
            parents,
        })
    }

    fn relu(&self) -> PyResult<Self> {
        let inner = self.inner.relu().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
        let requires_grad = self.requires_grad;
        let mut grad_fn = None;
        let mut parents = Vec::new();

        if requires_grad {
            grad_fn = Some(Arc::new(ReluNode { input: self.inner.clone() }) as Arc<dyn OpNode>);
            parents.push(self.clone());
        }

        Ok(PyTensor { 
            inner, 
            grad: if requires_grad { Some(Arc::new(Mutex::new(None))) } else { None },
            grad_fn,
            requires_grad,
            parents,
        })
    }

    fn sin(&self) -> PyResult<Self> {
        let inner = self.inner.sin().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
        let requires_grad = self.requires_grad;
        let mut grad_fn = None;
        let mut parents = Vec::new();

        if requires_grad {
            grad_fn = Some(Arc::new(SinNode { input: self.inner.clone() }) as Arc<dyn OpNode>);
            parents.push(self.clone());
        }

        Ok(PyTensor { 
            inner, 
            grad: if requires_grad { Some(Arc::new(Mutex::new(None))) } else { None },
            grad_fn,
            requires_grad,
            parents,
        })
    }

    fn cos(&self) -> PyResult<Self> {
        let inner = self.inner.cos().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
        let requires_grad = self.requires_grad;
        let mut grad_fn = None;
        let mut parents = Vec::new();

        if requires_grad {
            grad_fn = Some(Arc::new(CosNode { input: self.inner.clone() }) as Arc<dyn OpNode>);
            parents.push(self.clone());
        }

        Ok(PyTensor { 
            inner, 
            grad: if requires_grad { Some(Arc::new(Mutex::new(None))) } else { None },
            grad_fn,
            requires_grad,
            parents,
        })
    }

    fn recip(&self) -> PyResult<Self> {
        let inner = self.inner.recip().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
        let requires_grad = self.requires_grad;
        let mut grad_fn = None;
        let mut parents = Vec::new();

        if requires_grad {
            grad_fn = Some(Arc::new(RecipNode { input: self.inner.clone() }) as Arc<dyn OpNode>);
            parents.push(self.clone());
        }

        Ok(PyTensor { 
            inner, 
            grad: if requires_grad { Some(Arc::new(Mutex::new(None))) } else { None },
            grad_fn,
            requires_grad,
            parents,
        })
    }

    fn clone(&self) -> Self {
        PyTensor { 
            inner: self.inner.clone(),
            grad: self.grad.clone(),
            grad_fn: self.grad_fn.clone(),
            requires_grad: self.requires_grad,
            parents: self.parents.clone(),
        }
    }
}

// Helpers
impl PyTensor {
    fn broadcast_to_same_rank(&self, other: &Tensor) -> PyResult<(Tensor, Tensor)> {
        let lhs_shape = self.inner.dims();
        let rhs_shape = other.dims();
        let lhs_rank = lhs_shape.len();
        let rhs_rank = rhs_shape.len();

        if lhs_rank == rhs_rank {
            return Ok((self.inner.clone(), other.clone()));
        }

        if lhs_rank < rhs_rank {
            let mut new_shape = vec![1; rhs_rank - lhs_rank];
            new_shape.extend_from_slice(lhs_shape);
            let lhs = self.inner.reshape(new_shape)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("LHS reshape failed: {}", e)))?;
            Ok((lhs, other.clone()))
        } else {
            let mut new_shape = vec![1; lhs_rank - rhs_rank];
            new_shape.extend_from_slice(rhs_shape);
            let rhs = other.reshape(new_shape)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("RHS reshape failed: {}", e)))?;
            Ok((self.inner.clone(), rhs))
        }
    }
}

#[pymodule]
fn torch_candle_backend(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTensor>()?;
    Ok(())
}
