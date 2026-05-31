use std::collections::HashMap;
use pyo3::prelude::*;

#[pyclass]
#[derive(Clone, Debug)]
pub struct SSAValue {
    #[pyo3(get, set)]
    pub id: usize,
    #[pyo3(get, set)]
    pub dtype: String,
    #[pyo3(get, set)]
    pub shape: Vec<usize>,
    #[pyo3(get, set)]
    pub is_inplace: bool,
    #[pyo3(get, set)]
    pub is_alive: bool,
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct SSANode {
    #[pyo3(get, set)]
    pub op_name: String,
    #[pyo3(get, set)]
    pub inputs: Vec<usize>,
    #[pyo3(get, set)]
    pub outputs: Vec<usize>,
    #[pyo3(get, set)]
    pub attributes: HashMap<String, String>,
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct SSABlock {
    #[pyo3(get, set)]
    pub nodes: Vec<SSANode>,
}

#[pyclass]
pub struct SSACompiler {
    #[pyo3(get, set)]
    pub inputs: Vec<SSAValue>,
    #[pyo3(get, set)]
    pub outputs: Vec<SSAValue>,
    #[pyo3(get, set)]
    pub block: SSABlock,
    #[pyo3(get, set)]
    pub value_registry: HashMap<usize, SSAValue>,
    #[pyo3(get, set)]
    pub memory_map: HashMap<usize, u64>,
}

#[pymethods]
impl SSACompiler {
    #[new]
    pub fn new() -> Self {
        Self {
            inputs: Vec::new(),
            outputs: Vec::new(),
            block: SSABlock { nodes: Vec::new() },
            value_registry: HashMap::new(),
            memory_map: HashMap::new(),
        }
    }

    pub fn register_value(&mut self, id: usize, dtype: String, shape: Vec<usize>) {
        self.value_registry.insert(id, SSAValue {
            id,
            dtype,
            shape,
            is_inplace: false,
            is_alive: true,
        });
    }

    pub fn add_node(&mut self, op_name: String, inputs: Vec<usize>, outputs: Vec<usize>, attrs: HashMap<String, String>) {
        self.block.nodes.push(SSANode {
            op_name,
            inputs,
            outputs,
            attributes: attrs,
        });
    }

    pub fn add_input(&mut self, val_id: usize) {
        if let Some(val) = self.value_registry.get(&val_id) {
            self.inputs.push(val.clone());
        }
    }

    pub fn add_output(&mut self, val_id: usize) {
        if let Some(val) = self.value_registry.get(&val_id) {
            self.outputs.push(val.clone());
        }
    }

    pub fn compile_and_optimize(&mut self) -> PyResult<()> {
        let mut last_use: HashMap<usize, usize> = HashMap::new();
        
        for (node_idx, node) in self.block.nodes.iter().enumerate() {
            for &inp in &node.inputs {
                last_use.insert(inp, node_idx);
            }
        }

        let mut active_offsets: HashMap<u64, usize> = HashMap::new();
        let mut current_offset: u64 = 0;
        let mut free_offsets: Vec<u64> = Vec::new();

        for (node_idx, node) in self.block.nodes.iter_mut().enumerate() {
            let mut freed_offsets = Vec::new();
            for (&offset, &val_id) in &active_offsets {
                if let Some(&last_use_node) = last_use.get(&val_id) {
                    if last_use_node < node_idx {
                        freed_offsets.push(offset);
                    }
                }
            }
            for offset in freed_offsets {
                active_offsets.remove(&offset);
                free_offsets.push(offset);
            }

            let mut aliased = false;
            for &inp in &node.inputs {
                if let Some(&last_use_node) = last_use.get(&inp) {
                    if last_use_node == node_idx {
                        if !node.outputs.is_empty() {
                            let out_val = node.outputs[0];
                            if let Some(val) = self.value_registry.get_mut(&out_val) {
                                val.is_inplace = true;
                            }
                            
                            let mut input_offset = 0;
                            let mut found_offset = false;
                            for (&offset, &vid) in &active_offsets {
                                if vid == inp {
                                    input_offset = offset;
                                    found_offset = true;
                                    break;
                                }
                            }
                            
                            if found_offset {
                                self.memory_map.insert(out_val, input_offset);
                                active_offsets.insert(input_offset, out_val);
                                aliased = true;
                                break;
                            }
                        }
                    }
                }
            }

            if !aliased {
                for &out_val in &node.outputs {
                    let offset = if let Some(off) = free_offsets.pop() {
                        off
                    } else {
                        let off = current_offset;
                        current_offset += 256;
                        off
                    };
                    self.memory_map.insert(out_val, offset);
                    active_offsets.insert(offset, out_val);
                }
            }
        }
        
        Ok(())
    }
}

#[pyfunction]
pub fn compile_ast(py: Python<'_>, func: PyObject) -> PyResult<SSACompiler> {
    let inspect = py.import_bound("inspect")?;
    let ast = py.import_bound("ast")?;
    let textwrap = py.import_bound("textwrap")?;
    
    let source_py = inspect.call_method1("getsource", (func,))?;
    let source: String = source_py.extract()?;
    
    let dedented_py = textwrap.call_method1("dedent", (source,))?;
    let dedented_source: String = dedented_py.extract()?;
    
    let parsed = ast.call_method1("parse", (dedented_source,))?;
    
    let mut compiler = SSACompiler::new();
    
    let body = parsed.getattr("body")?;
    let body_list = body.downcast::<pyo3::types::PyList>()?;
    
    let mut next_val_id = 1;
    let mut var_map: HashMap<String, usize> = HashMap::new();
    
    for stmt in body_list.iter() {
        let class_name = stmt.getattr("__class__")?.getattr("__name__")?.extract::<String>()?;
        if class_name == "FunctionDef" {
            let func_body_val = stmt.getattr("body")?;
            let func_body = func_body_val.downcast::<pyo3::types::PyList>()?;
            
            for sub_stmt in func_body.iter() {
                let sub_class = sub_stmt.getattr("__class__")?.getattr("__name__")?.extract::<String>()?;
                if sub_class == "Assign" {
                    let value = sub_stmt.getattr("value")?;
                    let val_class = value.getattr("__class__")?.getattr("__name__")?.extract::<String>()?;
                    
                    if val_class == "BinOp" {
                        let left = value.getattr("left")?;
                        let right = value.getattr("right")?;
                        let op = value.getattr("op")?.getattr("__class__")?.getattr("__name__")?.extract::<String>()?;
                        
                        let left_name: String = if left.hasattr("id")? {
                            left.getattr("id")?.extract()?
                        } else {
                            "constant".to_string()
                        };
                        let right_name: String = if right.hasattr("id")? {
                            right.getattr("id")?.extract()?
                        } else {
                            "constant".to_string()
                        };
                        
                        let left_id = *var_map.entry(left_name).or_insert_with(|| {
                            let id = next_val_id;
                            next_val_id += 1;
                            compiler.register_value(id, "float32".to_string(), vec![1]);
                            id
                        });
                        
                        let right_id = *var_map.entry(right_name).or_insert_with(|| {
                            let id = next_val_id;
                            next_val_id += 1;
                            compiler.register_value(id, "float32".to_string(), vec![1]);
                            id
                        });
                        
                        let targets_val = sub_stmt.getattr("targets")?;
                        let targets = targets_val.downcast::<pyo3::types::PyList>()?;
                        let target_name: String = targets.get_item(0)?.getattr("id")?.extract()?;
                        let out_id = next_val_id;
                        next_val_id += 1;
                        var_map.insert(target_name, out_id);
                        compiler.register_value(out_id, "float32".to_string(), vec![1]);
                        
                        let mut attrs = HashMap::new();
                        attrs.insert("op".to_string(), op);
                        compiler.add_node("binop".to_string(), vec![left_id, right_id], vec![out_id], attrs);
                    }
                } else if sub_class == "If" {
                    let test = sub_stmt.getattr("test")?;
                    let test_class = test.getattr("__class__")?.getattr("__name__")?.extract::<String>()?;
                    if test_class == "Compare" {
                        let left = test.getattr("left")?;
                        let left_name: String = left.getattr("id")?.extract()?;
                        let left_id = *var_map.entry(left_name).or_insert(1);
                        
                        let mut attrs = HashMap::new();
                        attrs.insert("control_flow".to_string(), "if_branch".to_string());
                        
                        let if_body_val = sub_stmt.getattr("body")?;
                        let if_body = if_body_val.downcast::<pyo3::types::PyList>()?;
                        for body_stmt in if_body.iter() {
                            let b_class = body_stmt.getattr("__class__")?.getattr("__name__")?.extract::<String>()?;
                            if b_class == "Assign" {
                                let targets_val = body_stmt.getattr("targets")?;
                                let targets = targets_val.downcast::<pyo3::types::PyList>()?;
                                let target_name: String = targets.get_item(0)?.getattr("id")?.extract()?;
                                let val_id = *var_map.entry(target_name).or_insert(1);
                                compiler.add_node("if_true_assign".to_string(), vec![left_id], vec![val_id], attrs.clone());
                            }
                        }
                    }
                } else if sub_class == "For" {
                    let target = sub_stmt.getattr("target")?;
                    let target_name: String = target.getattr("id")?.extract()?;
                    let loop_var_id = *var_map.entry(target_name).or_insert(1);
                    
                    let mut attrs = HashMap::new();
                    attrs.insert("control_flow".to_string(), "for_loop".to_string());
                    
                    let for_body_val = sub_stmt.getattr("body")?;
                    let for_body = for_body_val.downcast::<pyo3::types::PyList>()?;
                    for body_stmt in for_body.iter() {
                        let b_class = body_stmt.getattr("__class__")?.getattr("__name__")?.extract::<String>()?;
                        if b_class == "Assign" {
                            let targets_val = body_stmt.getattr("targets")?;
                            let targets = targets_val.downcast::<pyo3::types::PyList>()?;
                            let target_name: String = targets.get_item(0)?.getattr("id")?.extract()?;
                            let val_id = *var_map.entry(target_name).or_insert(1);
                            compiler.add_node("for_loop_body_assign".to_string(), vec![loop_var_id], vec![val_id], attrs.clone());
                        }
                    }
                }
            }
        }
    }
    
    compiler.compile_and_optimize()?;
    Ok(compiler)
}
