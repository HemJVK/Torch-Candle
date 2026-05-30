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
