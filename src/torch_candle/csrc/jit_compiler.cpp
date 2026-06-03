#include <torch/extension.h>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <sstream>
#include <cctype>
#include <cmath>
#include <algorithm>

enum TokenType {
    TOKEN_EOF,
    TOKEN_IDENT,
    TOKEN_NUMBER,
    TOKEN_PLUS,
    TOKEN_MINUS,
    TOKEN_MUL,
    TOKEN_DIV,
    TOKEN_POW,
    TOKEN_LPAREN,
    TOKEN_RPAREN
};

struct Token {
    TokenType type;
    std::string text;
};

class Lexer {
    std::string src;
    size_t pos = 0;
public:
    Lexer(std::string src) : src(src) {}
    Token next() {
        while (pos < src.size() && std::isspace(src[pos])) {
            pos++;
        }
        if (pos >= src.size()) return {TOKEN_EOF, ""};
        char c = src[pos];
        if (c == '+') { pos++; return {TOKEN_PLUS, "+"}; }
        if (c == '-') { pos++; return {TOKEN_MINUS, "-"}; }
        if (c == '*') {
            pos++;
            if (pos < src.size() && src[pos] == '*') {
                pos++;
                return {TOKEN_POW, "**"};
            }
            return {TOKEN_MUL, "*"};
        }
        if (c == '/') { pos++; return {TOKEN_DIV, "/"}; }
        if (c == '(') { pos++; return {TOKEN_LPAREN, "("}; }
        if (c == ')') { pos++; return {TOKEN_RPAREN, ")"}; }
        if (std::isdigit(c) || c == '.') {
            std::string s;
            while (pos < src.size() && (std::isdigit(src[pos]) || src[pos] == '.')) {
                s += src[pos++];
            }
            return {TOKEN_NUMBER, s};
        }
        if (std::isalpha(c) || c == '_') {
            std::string s;
            while (pos < src.size() && (std::isalnum(src[pos]) || src[pos] == '_')) {
                s += src[pos++];
            }
            return {TOKEN_IDENT, s};
        }
        pos++;
        return {TOKEN_EOF, ""};
    }
};

struct SSAInstruction {
    std::string result;
    std::string op; // "add", "sub", "mul", "div", "pow", "const", "ident"
    std::string arg1;
    std::string arg2;
    double const_val = 0.0;
    bool has_const = false;
};

class SSACompiler;

class Parser {
    Lexer lexer;
    Token curr;
    SSACompiler* compiler;
    void consume() { curr = lexer.next(); }
public:
    Parser(std::string src, SSACompiler* comp);
    std::string parse();
private:
    std::string parse_expr();
    std::string parse_term();
    std::string parse_factor();
    std::string parse_primary();
};

class SSACompiler {
public:
    std::vector<std::string> inputs;
    std::string output;
    std::vector<SSAInstruction> instructions;
    int var_counter = 0;

    SSACompiler() {}

    std::string next_var() {
        return "%v" + std::to_string(var_counter++);
    }

    void compile(const std::string& expr) {
        Parser parser(expr, this);
        output = parser.parse();
    }

    at::Tensor execute(const std::vector<at::Tensor>& input_tensors, const std::vector<std::string>& input_names) {
        std::unordered_map<std::string, at::Tensor> env;
        for (size_t i = 0; i < input_tensors.size() && i < input_names.size(); ++i) {
            env[input_names[i]] = input_tensors[i];
        }

        for (auto& inst : instructions) {
            if (inst.op == "const") {
                env[inst.result] = torch::tensor(inst.const_val, torch::dtype(torch::kFloat32));
            } else if (inst.op == "ident") {
                auto it = env.find(inst.arg1);
                if (it == env.end()) throw std::runtime_error("SSA execution: variable not found: " + inst.arg1);
                env[inst.result] = it->second;
            } else {
                at::Tensor a1, a2;
                if (inst.has_const) {
                    auto it1 = env.find(inst.arg1);
                    if (it1 == env.end()) throw std::runtime_error("SSA execution: variable not found: " + inst.arg1);
                    a1 = it1->second;
                    a2 = torch::tensor(inst.const_val, torch::dtype(torch::kFloat32));
                } else {
                    auto it1 = env.find(inst.arg1);
                    if (it1 == env.end()) throw std::runtime_error("SSA execution: variable not found: " + inst.arg1);
                    auto it2 = env.find(inst.arg2);
                    if (it2 == env.end()) throw std::runtime_error("SSA execution: variable not found: " + inst.arg2);
                    a1 = it1->second;
                    a2 = it2->second;
                }

                if (inst.op == "add") {
                    env[inst.result] = a1 + a2;
                } else if (inst.op == "sub") {
                    env[inst.result] = a1 - a2;
                } else if (inst.op == "mul") {
                    env[inst.result] = a1 * a2;
                } else if (inst.op == "div") {
                    env[inst.result] = a1 / a2;
                } else if (inst.op == "pow") {
                    env[inst.result] = a1.pow(a2);
                } else {
                    throw std::runtime_error("Unknown op: " + inst.op);
                }
            }
        }
        auto it = env.find(output);
        if (it == env.end()) throw std::runtime_error("SSA execution: output not found: " + output);
        return it->second;
    }

    SSACompiler compose_jacrev(const std::string& target_var) {
        SSACompiler derivative_compiler;
        derivative_compiler.inputs = this->inputs;
        derivative_compiler.inputs.push_back("grad_output");

        derivative_compiler.instructions = this->instructions;
        derivative_compiler.var_counter = this->var_counter;

        std::unordered_map<std::string, std::string> d_map;
        d_map[this->output] = "grad_output";

        auto get_or_create_d = [&](const std::string& var_name) -> std::string {
            auto it = d_map.find(var_name);
            if (it != d_map.end()) return it->second;
            std::string d_var = "%d_" + var_name.substr(1);
            SSAInstruction inst;
            inst.result = d_var;
            inst.op = "const";
            inst.const_val = 0.0;
            derivative_compiler.instructions.push_back(inst);
            d_map[var_name] = d_var;
            return d_var;
        };

        get_or_create_d(target_var);

        for (int i = (int)this->instructions.size() - 1; i >= 0; --i) {
            auto& inst = this->instructions[i];
            if (inst.op == "const" || inst.op == "ident") {
                continue;
            }
            std::string d_res = get_or_create_d(inst.result);

            if (inst.op == "add") {
                if (inst.has_const) {
                    std::string d_arg1 = get_or_create_d(inst.arg1);
                    std::string next_d = derivative_compiler.next_var();
                    SSAInstruction add_inst;
                    add_inst.result = next_d;
                    add_inst.op = "add";
                    add_inst.arg1 = d_arg1;
                    add_inst.arg2 = d_res;
                    derivative_compiler.instructions.push_back(add_inst);
                    d_map[inst.arg1] = next_d;
                } else {
                    std::string d_arg1 = get_or_create_d(inst.arg1);
                    std::string d_arg2 = get_or_create_d(inst.arg2);

                    std::string next_d1 = derivative_compiler.next_var();
                    SSAInstruction add_inst1;
                    add_inst1.result = next_d1;
                    add_inst1.op = "add";
                    add_inst1.arg1 = d_arg1;
                    add_inst1.arg2 = d_res;
                    derivative_compiler.instructions.push_back(add_inst1);
                    d_map[inst.arg1] = next_d1;

                    std::string next_d2 = derivative_compiler.next_var();
                    SSAInstruction add_inst2;
                    add_inst2.result = next_d2;
                    add_inst2.op = "add";
                    add_inst2.arg1 = d_arg2;
                    add_inst2.arg2 = d_res;
                    derivative_compiler.instructions.push_back(add_inst2);
                    d_map[inst.arg2] = next_d2;
                }
            } else if (inst.op == "sub") {
                if (inst.has_const) {
                    std::string d_arg1 = get_or_create_d(inst.arg1);
                    std::string next_d = derivative_compiler.next_var();
                    SSAInstruction add_inst;
                    add_inst.result = next_d;
                    add_inst.op = "add";
                    add_inst.arg1 = d_arg1;
                    add_inst.arg2 = d_res;
                    derivative_compiler.instructions.push_back(add_inst);
                    d_map[inst.arg1] = next_d;
                } else {
                    std::string d_arg1 = get_or_create_d(inst.arg1);
                    std::string d_arg2 = get_or_create_d(inst.arg2);

                    std::string next_d1 = derivative_compiler.next_var();
                    SSAInstruction add_inst1;
                    add_inst1.result = next_d1;
                    add_inst1.op = "add";
                    add_inst1.arg1 = d_arg1;
                    add_inst1.arg2 = d_res;
                    derivative_compiler.instructions.push_back(add_inst1);
                    d_map[inst.arg1] = next_d1;

                    std::string next_d2 = derivative_compiler.next_var();
                    SSAInstruction add_inst2;
                    add_inst2.result = next_d2;
                    add_inst2.op = "sub";
                    add_inst2.arg1 = d_arg2;
                    add_inst2.arg2 = d_res;
                    derivative_compiler.instructions.push_back(add_inst2);
                    d_map[inst.arg2] = next_d2;
                }
            } else if (inst.op == "mul") {
                if (inst.has_const) {
                    std::string d_arg1 = get_or_create_d(inst.arg1);
                    std::string temp = derivative_compiler.next_var();
                    SSAInstruction mul_c;
                    mul_c.result = temp;
                    mul_c.op = "mul";
                    mul_c.arg1 = d_res;
                    mul_c.has_const = true;
                    mul_c.const_val = inst.const_val;
                    derivative_compiler.instructions.push_back(mul_c);

                    std::string next_d = derivative_compiler.next_var();
                    SSAInstruction add_inst;
                    add_inst.result = next_d;
                    add_inst.op = "add";
                    add_inst.arg1 = d_arg1;
                    add_inst.arg2 = temp;
                    derivative_compiler.instructions.push_back(add_inst);
                    d_map[inst.arg1] = next_d;
                } else {
                    std::string d_arg1 = get_or_create_d(inst.arg1);
                    std::string d_arg2 = get_or_create_d(inst.arg2);

                    std::string temp1 = derivative_compiler.next_var();
                    SSAInstruction mul_inst1;
                    mul_inst1.result = temp1;
                    mul_inst1.op = "mul";
                    mul_inst1.arg1 = d_res;
                    mul_inst1.arg2 = inst.arg2;
                    derivative_compiler.instructions.push_back(mul_inst1);

                    std::string next_d1 = derivative_compiler.next_var();
                    SSAInstruction add_inst1;
                    add_inst1.result = next_d1;
                    add_inst1.op = "add";
                    add_inst1.arg1 = d_arg1;
                    add_inst1.arg2 = temp1;
                    derivative_compiler.instructions.push_back(add_inst1);
                    d_map[inst.arg1] = next_d1;

                    std::string temp2 = derivative_compiler.next_var();
                    SSAInstruction mul_inst2;
                    mul_inst2.result = temp2;
                    mul_inst2.op = "mul";
                    mul_inst2.arg1 = d_res;
                    mul_inst2.arg2 = inst.arg1;
                    derivative_compiler.instructions.push_back(mul_inst2);

                    std::string next_d2 = derivative_compiler.next_var();
                    SSAInstruction add_inst2;
                    add_inst2.result = next_d2;
                    add_inst2.op = "add";
                    add_inst2.arg1 = d_arg2;
                    add_inst2.arg2 = temp2;
                    derivative_compiler.instructions.push_back(add_inst2);
                    d_map[inst.arg2] = next_d2;
                }
            }
        }

        derivative_compiler.output = d_map[target_var];
        return derivative_compiler;
    }
};

Parser::Parser(std::string src, SSACompiler* comp) : lexer(src), compiler(comp) { consume(); }

std::string Parser::parse() {
    return parse_expr();
}

std::string Parser::parse_expr() {
    auto node = parse_term();
    while (curr.type == TOKEN_PLUS || curr.type == TOKEN_MINUS) {
        auto type = curr.type;
        consume();
        auto right = parse_term();
        std::string res = compiler->next_var();
        SSAInstruction inst;
        inst.result = res;
        inst.op = (type == TOKEN_PLUS) ? "add" : "sub";
        
        if (right[0] != '%') {
            inst.arg1 = node;
            inst.has_const = true;
            inst.const_val = std::stod(right);
        } else if (node[0] != '%') {
            inst.arg1 = right;
            inst.has_const = true;
            inst.const_val = std::stod(node);
        } else {
            inst.arg1 = node;
            inst.arg2 = right;
        }
        compiler->instructions.push_back(inst);
        node = res;
    }
    return node;
}

std::string Parser::parse_term() {
    auto node = parse_factor();
    while (curr.type == TOKEN_MUL || curr.type == TOKEN_DIV) {
        auto type = curr.type;
        consume();
        auto right = parse_factor();
        std::string res = compiler->next_var();
        SSAInstruction inst;
        inst.result = res;
        inst.op = (type == TOKEN_MUL) ? "mul" : "div";
        
        if (right[0] != '%') {
            inst.arg1 = node;
            inst.has_const = true;
            inst.const_val = std::stod(right);
        } else if (node[0] != '%') {
            inst.arg1 = right;
            inst.has_const = true;
            inst.const_val = std::stod(node);
        } else {
            inst.arg1 = node;
            inst.arg2 = right;
        }
        compiler->instructions.push_back(inst);
        node = res;
    }
    return node;
}

std::string Parser::parse_factor() {
    auto node = parse_primary();
    while (curr.type == TOKEN_POW) {
        consume();
        auto right = parse_primary();
        std::string res = compiler->next_var();
        SSAInstruction inst;
        inst.result = res;
        inst.op = "pow";
        if (right[0] != '%') {
            inst.arg1 = node;
            inst.has_const = true;
            inst.const_val = std::stod(right);
        } else if (node[0] != '%') {
            inst.arg1 = right;
            inst.has_const = true;
            inst.const_val = std::stod(node);
        } else {
            inst.arg1 = node;
            inst.arg2 = right;
        }
        compiler->instructions.push_back(inst);
        node = res;
    }
    return node;
}

std::string Parser::parse_primary() {
    if (curr.type == TOKEN_NUMBER) {
        std::string val = curr.text;
        consume();
        return val;
    }
    if (curr.type == TOKEN_IDENT) {
        std::string name = curr.text;
        consume();
        std::string res = compiler->next_var();
        SSAInstruction inst;
        inst.result = res;
        inst.op = "ident";
        inst.arg1 = name;
        compiler->instructions.push_back(inst);
        if (std::find(compiler->inputs.begin(), compiler->inputs.end(), name) == compiler->inputs.end()) {
            compiler->inputs.push_back(name);
        }
        return res;
    }
    if (curr.type == TOKEN_LPAREN) {
        consume();
        auto node = parse_expr();
        if (curr.type != TOKEN_RPAREN) throw std::runtime_error("Expected ')'");
        consume();
        return node;
    }
    throw std::runtime_error("Unexpected token: " + curr.text);
}

class JITCompiledFunction {
    SSACompiler compiler;
    std::string expr_str;
public:
    JITCompiledFunction(std::string expr) : expr_str(expr) {
        compiler.compile(expr);
    }
    
    at::Tensor forward(const std::vector<at::Tensor>& inputs, const std::vector<std::string>& input_names) {
        return compiler.execute(inputs, input_names);
    }
    
    std::vector<at::Tensor> backward(
        const std::vector<at::Tensor>& inputs,
        const std::vector<std::string>& input_names,
        at::Tensor grad_output
    ) {
        std::vector<at::Tensor> grads;
        for (const auto& name : input_names) {
            SSACompiler derivative = compiler.compose_jacrev(name);
            std::vector<at::Tensor> derivative_inputs = inputs;
            derivative_inputs.push_back(grad_output);
            std::vector<std::string> derivative_input_names = input_names;
            derivative_input_names.push_back("grad_output");
            grads.push_back(derivative.execute(derivative_inputs, derivative_input_names));
        }
        return grads;
    }
};

class BatchTensor0_cpp {
public:
    at::Tensor value;
    int batch_dim;
    BatchTensor0_cpp(at::Tensor val, int dim) : value(val), batch_dim(dim) {}
    
    BatchTensor0_cpp add(const BatchTensor0_cpp& other) {
        return BatchTensor0_cpp(value + other.value, batch_dim);
    }
    BatchTensor0_cpp add_const(double val) {
        return BatchTensor0_cpp(value + val, batch_dim);
    }
    BatchTensor0_cpp sub(const BatchTensor0_cpp& other) {
        return BatchTensor0_cpp(value - other.value, batch_dim);
    }
    BatchTensor0_cpp sub_const(double val) {
        return BatchTensor0_cpp(value - val, batch_dim);
    }
    BatchTensor0_cpp mul(const BatchTensor0_cpp& other) {
        return BatchTensor0_cpp(value * other.value, batch_dim);
    }
    BatchTensor0_cpp mul_const(double val) {
        return BatchTensor0_cpp(value * val, batch_dim);
    }
    BatchTensor0_cpp div(const BatchTensor0_cpp& other) {
        return BatchTensor0_cpp(value / other.value, batch_dim);
    }
    BatchTensor0_cpp div_const(double val) {
        return BatchTensor0_cpp(value / val, batch_dim);
    }
};

class BatchTensor1_cpp {
public:
    at::Tensor value;
    int batch_dim;
    BatchTensor1_cpp(at::Tensor val, int dim) : value(val), batch_dim(dim) {}
    
    BatchTensor1_cpp add(const BatchTensor1_cpp& other) {
        return BatchTensor1_cpp(value + other.value, batch_dim);
    }
    BatchTensor1_cpp add_const(double val) {
        return BatchTensor1_cpp(value + val, batch_dim);
    }
    BatchTensor1_cpp sub(const BatchTensor1_cpp& other) {
        return BatchTensor1_cpp(value - other.value, batch_dim);
    }
    BatchTensor1_cpp sub_const(double val) {
        return BatchTensor1_cpp(value - val, batch_dim);
    }
    BatchTensor1_cpp mul(const BatchTensor1_cpp& other) {
        return BatchTensor1_cpp(value * other.value, batch_dim);
    }
    BatchTensor1_cpp mul_const(double val) {
        return BatchTensor1_cpp(value * val, batch_dim);
    }
    BatchTensor1_cpp div(const BatchTensor1_cpp& other) {
        return BatchTensor1_cpp(value / other.value, batch_dim);
    }
    BatchTensor1_cpp div_const(double val) {
        return BatchTensor1_cpp(value / val, batch_dim);
    }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<JITCompiledFunction>(m, "JITCompiledFunction")
        .def(py::init<std::string>())
        .def("forward", &JITCompiledFunction::forward, py::call_guard<py::gil_scoped_release>())
        .def("backward", &JITCompiledFunction::backward, py::call_guard<py::gil_scoped_release>());

    py::class_<BatchTensor0_cpp>(m, "BatchTensor0_cpp")
        .def(py::init<at::Tensor, int>())
        .def_readwrite("value", &BatchTensor0_cpp::value)
        .def_readwrite("batch_dim", &BatchTensor0_cpp::batch_dim)
        .def("add", &BatchTensor0_cpp::add)
        .def("add_const", &BatchTensor0_cpp::add_const)
        .def("sub", &BatchTensor0_cpp::sub)
        .def("sub_const", &BatchTensor0_cpp::sub_const)
        .def("mul", &BatchTensor0_cpp::mul)
        .def("mul_const", &BatchTensor0_cpp::mul_const)
        .def("div", &BatchTensor0_cpp::div)
        .def("div_const", &BatchTensor0_cpp::div_const);

    py::class_<BatchTensor1_cpp>(m, "BatchTensor1_cpp")
        .def(py::init<at::Tensor, int>())
        .def_readwrite("value", &BatchTensor1_cpp::value)
        .def_readwrite("batch_dim", &BatchTensor1_cpp::batch_dim)
        .def("add", &BatchTensor1_cpp::add)
        .def("add_const", &BatchTensor1_cpp::add_const)
        .def("sub", &BatchTensor1_cpp::sub)
        .def("sub_const", &BatchTensor1_cpp::sub_const)
        .def("mul", &BatchTensor1_cpp::mul)
        .def("mul_const", &BatchTensor1_cpp::mul_const)
        .def("div", &BatchTensor1_cpp::div)
        .def("div_const", &BatchTensor1_cpp::div_const);
}
