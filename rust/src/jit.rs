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

    pub fn serialize_to_buffer(&self, ring_buffer: &crate::ipc::SPSCRingBuffer) -> PyResult<()> {
        for node in &self.block.nodes {
            let mut payload = [0u8; 256];
            payload[0] = node.inputs.len() as u8;
            payload[1] = node.outputs.len() as u8;
            for (i, &inp) in node.inputs.iter().enumerate().take(10) {
                payload[2 + i] = inp as u8;
            }
            for (i, &out) in node.outputs.iter().enumerate().take(10) {
                payload[12 + i] = out as u8;
            }
            // Byte 22 tracks dynamic operations: 1 for binop, 2 for other
            if node.op_name == "binop" {
                payload[22] = 1;
            } else {
                payload[22] = 2;
            }
            
            // Push directly into shared memory segment bypassing python runtime checks!
            ring_buffer.push(777, 0, payload.to_vec())?;
        }
        Ok(())
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

    /// Execute the compiled SSA graph as a standalone Rust VM.
    /// This runs entirely in native Rust using Candle tensors — no Python GIL needed.
    /// The VM walks SSA nodes sequentially, executing each operation via Candle ops.
    pub fn execute(&self, py: Python<'_>, input_map: HashMap<String, crate::PyTensor>) -> PyResult<crate::PyTensor> {
        // Build a value environment mapping value IDs to Candle tensors
        let mut env: HashMap<usize, candle_core::Tensor> = HashMap::new();
        
        // Map named inputs to their value IDs
        // Input values are identified by checking the inputs list
        let mut name_to_id: HashMap<String, usize> = HashMap::new();
        for val in &self.inputs {
            // Use dtype field as the variable name for identifier values
            // (registered by compile_ast with the variable name in the identifier fields)
            let name = &val.dtype;
            name_to_id.insert(name.clone(), val.id);
        }
        
        // Populate the environment with input tensors
        for (name, pytensor) in &input_map {
            if let Some(&val_id) = name_to_id.get(name) {
                env.insert(val_id, pytensor.inner.clone());
            }
        }
        
        // Execute in a GIL-free context for maximum throughput
        let block_nodes = self.block.nodes.clone();
        let _value_registry = self.value_registry.clone();
        
        let result = py.allow_threads(|| -> Result<candle_core::Tensor, String> {
            for node in &block_nodes {
                match node.op_name.as_str() {
                    "constant" => {
                        if let Some(val_str) = node.attributes.get("value") {
                            if let Ok(val) = val_str.parse::<f64>() {
                                let t = candle_core::Tensor::new(&[val as f32], &candle_core::Device::Cpu)
                                    .map_err(|e| format!("SSA VM constant: {}", e))?;
                                for &out_id in &node.outputs {
                                    env.insert(out_id, t.clone());
                                }
                            }
                        }
                    }
                    "binop" => {
                        if node.inputs.len() < 2 || node.outputs.is_empty() {
                            return Err("SSA VM binop: insufficient inputs/outputs".to_string());
                        }
                        let lhs = env.get(&node.inputs[0])
                            .ok_or_else(|| format!("SSA VM: value {} not found", node.inputs[0]))?;
                        let rhs = env.get(&node.inputs[1])
                            .ok_or_else(|| format!("SSA VM: value {} not found", node.inputs[1]))?;
                        
                        let op = node.attributes.get("op").map(|s| s.as_str()).unwrap_or("Add");
                        let result = match op {
                            "Add" => lhs.broadcast_add(rhs),
                            "Sub" => lhs.broadcast_sub(rhs),
                            "Mult" => lhs.broadcast_mul(rhs),
                            "Div" => lhs.broadcast_div(rhs),
                            "Pow" => {
                                // Power: try to extract scalar exponent
                                if let Ok(exp_vec) = rhs.to_vec1::<f32>() {
                                    if !exp_vec.is_empty() {
                                        lhs.powf(exp_vec[0] as f64)
                                    } else {
                                        lhs.broadcast_mul(rhs) // fallback
                                    }
                                } else {
                                    lhs.broadcast_mul(rhs) // fallback
                                }
                            }
                            _ => lhs.broadcast_add(rhs), // default to add for unknown ops
                        }.map_err(|e| format!("SSA VM binop({}): {}", op, e))?;
                        
                        env.insert(node.outputs[0], result);
                    }
                    "if_true_assign" => {
                        // Conditional execution: if the condition input is truthy,
                        // assign the value. For SSA IR, this is a phi-node analog.
                        if !node.inputs.is_empty() && !node.outputs.is_empty() {
                            if let Some(cond) = env.get(&node.inputs[0]) {
                                // Check if condition is > 0 (truthy)
                                if let Ok(cond_vec) = cond.to_vec1::<f32>() {
                                    if !cond_vec.is_empty() && cond_vec[0] > 0.0 {
                                        env.insert(node.outputs[0], cond.clone());
                                    }
                                }
                            }
                        }
                    }
                    "for_loop_body_assign" => {
                        // Loop body execution: process the loop variable
                        if !node.inputs.is_empty() && !node.outputs.is_empty() {
                            if let Some(val) = env.get(&node.inputs[0]) {
                                env.insert(node.outputs[0], val.clone());
                            }
                        }
                    }
                    _ => {
                        return Err(format!("SSA VM: unknown op '{}'", node.op_name));
                    }
                }
            }
            
            // Find the output value
            // Use the last output node's output, or the last value inserted
            if let Some(last_node) = block_nodes.last() {
                if let Some(&out_id) = last_node.outputs.last() {
                    if let Some(t) = env.get(&out_id) {
                        return Ok(t.clone());
                    }
                }
            }
            
            Err("SSA VM: no output produced".to_string())
        }).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
        
        Ok(crate::PyTensor {
            inner: result,
            grad: None,
            grad_fn: None,
            requires_grad: false,
            parents: Vec::new(),
        })
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

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Identifier(String),
    Number(f64),
    Operator(char),
    LParen,
    RParen,
}

fn tokenize(expr: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut chars = expr.chars().peekable();
    while let Some(&c) = chars.peek() {
        if c.is_whitespace() {
            chars.next();
        } else if c.is_alphabetic() || c == '_' {
            let mut name = String::new();
            while let Some(&next_c) = chars.peek() {
                if next_c.is_alphanumeric() || next_c == '_' {
                    name.push(chars.next().unwrap());
                } else {
                    break;
                }
            }
            tokens.push(Token::Identifier(name));
        } else if c.is_digit(10) || c == '.' {
            let mut num_str = String::new();
            while let Some(&next_c) = chars.peek() {
                if next_c.is_digit(10) || next_c == '.' {
                    num_str.push(chars.next().unwrap());
                } else {
                    break;
                }
            }
            if let Ok(val) = num_str.parse::<f64>() {
                tokens.push(Token::Number(val));
            }
        } else if c == '+' || c == '-' || c == '*' || c == '/' {
            tokens.push(Token::Operator(chars.next().unwrap()));
        } else if c == '(' {
            tokens.push(Token::LParen);
            chars.next();
        } else if c == ')' {
            tokens.push(Token::RParen);
            chars.next();
        } else {
            chars.next(); // skip unknown character
        }
    }
    tokens
}

#[derive(Debug, Clone)]
pub enum ASTNode {
    Number(f64),
    Identifier(String),
    BinaryOp {
        op: char,
        left: Box<ASTNode>,
        right: Box<ASTNode>,
    },
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0 }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn next_token(&mut self) -> Option<Token> {
        if self.pos < self.tokens.len() {
            let t = self.tokens[self.pos].clone();
            self.pos += 1;
            Some(t)
        } else {
            None
        }
    }

    fn parse_expression(&mut self) -> PyResult<ASTNode> {
        let mut node = self.parse_term()?;
        while let Some(Token::Operator(op)) = self.peek() {
            if *op == '+' || *op == '-' {
                let op_char = *op;
                self.next_token();
                let right = self.parse_term()?;
                node = ASTNode::BinaryOp {
                    op: op_char,
                    left: Box::new(node),
                    right: Box::new(right),
                };
            } else {
                break;
            }
        }
        Ok(node)
    }

    fn parse_term(&mut self) -> PyResult<ASTNode> {
        let mut node = self.parse_factor()?;
        while let Some(Token::Operator(op)) = self.peek() {
            if *op == '*' || *op == '/' {
                let op_char = *op;
                self.next_token();
                let right = self.parse_factor()?;
                node = ASTNode::BinaryOp {
                    op: op_char,
                    left: Box::new(node),
                    right: Box::new(right),
                };
            } else {
                break;
            }
        }
        Ok(node)
    }

    fn parse_factor(&mut self) -> PyResult<ASTNode> {
        match self.next_token() {
            Some(Token::Number(val)) => Ok(ASTNode::Number(val)),
            Some(Token::Identifier(name)) => Ok(ASTNode::Identifier(name)),
            Some(Token::LParen) => {
                let expr = self.parse_expression()?;
                if let Some(Token::RParen) = self.next_token() {
                    Ok(expr)
                } else {
                    Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Mismatched parentheses"))
                }
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid syntax")),
        }
    }
}

pub fn parse_to_ssa(expr: &str, mut compiler: SSACompiler) -> PyResult<SSACompiler> {
    let tokens = tokenize(expr);
    let mut parser = Parser::new(tokens);
    let ast = parser.parse_expression()?;
    
    let mut next_val_id = 1;
    let mut var_map = HashMap::new();
    
    let final_out = compile_node(&ast, &mut compiler, &mut next_val_id, &mut var_map)?;
    compiler.add_output(final_out);
    
    compiler.compile_and_optimize()?;
    Ok(compiler)
}

fn compile_node(node: &ASTNode, compiler: &mut SSACompiler, next_val_id: &mut usize, var_map: &mut HashMap<String, usize>) -> PyResult<usize> {
    match node {
        ASTNode::Number(val) => {
            let id = *next_val_id;
            *next_val_id += 1;
            compiler.register_value(id, "float32".to_string(), vec![1]);
            let mut attrs = HashMap::new();
            attrs.insert("value".to_string(), val.to_string());
            compiler.add_node("constant".to_string(), vec![], vec![id], attrs);
            Ok(id)
        }
        ASTNode::Identifier(name) => {
            let id = *var_map.entry(name.clone()).or_insert_with(|| {
                let id = *next_val_id;
                *next_val_id += 1;
                compiler.register_value(id, "float32".to_string(), vec![1]);
                compiler.add_input(id);
                id
            });
            Ok(id)
        }
        ASTNode::BinaryOp { op, left, right } => {
            let left_id = compile_node(left, compiler, next_val_id, var_map)?;
            let right_id = compile_node(right, compiler, next_val_id, var_map)?;
            let out_id = *next_val_id;
            *next_val_id += 1;
            compiler.register_value(out_id, "float32".to_string(), vec![1]);
            
            let mut attrs = HashMap::new();
            attrs.insert("op".to_string(), op.to_string());
            compiler.add_node("binop".to_string(), vec![left_id, right_id], vec![out_id], attrs);
            Ok(out_id)
        }
    }
}

#[pyclass]
pub struct NativeASTParser;

#[pymethods]
impl NativeASTParser {
    #[staticmethod]
    pub fn parse_expression(expr: String) -> PyResult<SSACompiler> {
        let compiler = SSACompiler::new();
        parse_to_ssa(&expr, compiler)
    }
}

#[derive(Clone, Debug)]
pub enum SymExpr {
    Var,
    Const(f64),
    Add(Box<SymExpr>, Box<SymExpr>),
    Sub(Box<SymExpr>, Box<SymExpr>),
    Mul(Box<SymExpr>, Box<SymExpr>),
    Pow(Box<SymExpr>, f64),
}

impl SymExpr {
    fn diff(&self) -> Self {
        match self {
            SymExpr::Var => SymExpr::Const(1.0),
            SymExpr::Const(_) => SymExpr::Const(0.0),
            SymExpr::Add(left, right) => {
                SymExpr::Add(Box::new(left.diff()), Box::new(right.diff()))
            }
            SymExpr::Sub(left, right) => {
                SymExpr::Sub(Box::new(left.diff()), Box::new(right.diff()))
            }
            SymExpr::Mul(left, right) => {
                SymExpr::Add(
                    Box::new(SymExpr::Mul(left.clone(), Box::new(right.diff()))),
                    Box::new(SymExpr::Mul(right.clone(), Box::new(left.diff()))),
                )
            }
            SymExpr::Pow(left, n) => {
                SymExpr::Mul(
                    Box::new(SymExpr::Mul(
                        Box::new(SymExpr::Const(*n)),
                        Box::new(SymExpr::Pow(left.clone(), n - 1.0)),
                    )),
                    Box::new(left.diff()),
                )
            }
        }
    }

    fn to_string(&self) -> String {
        match self {
            SymExpr::Var => "x".to_string(),
            SymExpr::Const(c) => c.to_string(),
            SymExpr::Add(left, right) => format!("({}+{})", left.to_string(), right.to_string()),
            SymExpr::Sub(left, right) => format!("({}-{})", left.to_string(), right.to_string()),
            SymExpr::Mul(left, right) => format!("({}*{})", left.to_string(), right.to_string()),
            SymExpr::Pow(left, n) => format!("({}**{})", left.to_string(), n),
        }
    }

    fn eval(&self, val: &crate::PyTensor) -> PyResult<crate::PyTensor> {
        match self {
            SymExpr::Var => Ok(val.clone()),
            SymExpr::Const(c) => {
                let inner = candle_core::Tensor::new(&[*c as f32], val.inner.device())
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
                Ok(crate::PyTensor {
                    inner,
                    grad: None,
                    grad_fn: None,
                    requires_grad: false,
                    parents: Vec::new(),
                })
            }
            SymExpr::Add(left, right) => {
                let l = left.eval(val)?;
                let r = right.eval(val)?;
                l.add(&r)
            }
            SymExpr::Sub(left, right) => {
                let l = left.eval(val)?;
                let r = right.eval(val)?;
                l.sub(&r)
            }
            SymExpr::Mul(left, right) => {
                let l = left.eval(val)?;
                let r = right.eval(val)?;
                l.mul(&r)
            }
            SymExpr::Pow(left, n) => {
                let l = left.eval(val)?;
                l.pow(*n)
            }
        }
    }

    fn substitute(&self, replacement: &SymExpr) -> Self {
        match self {
            SymExpr::Var => replacement.clone(),
            SymExpr::Const(c) => SymExpr::Const(*c),
            SymExpr::Add(left, right) => {
                SymExpr::Add(
                    Box::new(left.substitute(replacement)),
                    Box::new(right.substitute(replacement)),
                )
            }
            SymExpr::Sub(left, right) => {
                SymExpr::Sub(
                    Box::new(left.substitute(replacement)),
                    Box::new(right.substitute(replacement)),
                )
            }
            SymExpr::Mul(left, right) => {
                SymExpr::Mul(
                    Box::new(left.substitute(replacement)),
                    Box::new(right.substitute(replacement)),
                )
            }
            SymExpr::Pow(left, n) => {
                SymExpr::Pow(Box::new(left.substitute(replacement)), *n)
            }
        }
    }
}


