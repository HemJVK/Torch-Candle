#include <torch/extension.h>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <sstream>
#include <cctype>
#include <cmath>

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

struct ASTNode {
    virtual ~ASTNode() = default;
    virtual at::Tensor eval(const std::unordered_map<std::string, at::Tensor>& env) = 0;
};

struct VarNode : public ASTNode {
    std::string name;
    VarNode(std::string name) : name(name) {}
    at::Tensor eval(const std::unordered_map<std::string, at::Tensor>& env) override {
        auto it = env.find(name);
        if (it == env.end()) throw std::runtime_error("Variable not found: " + name);
        return it->second;
    }
};

struct ConstNode : public ASTNode {
    double val;
    ConstNode(double val) : val(val) {}
    at::Tensor eval(const std::unordered_map<std::string, at::Tensor>& env) override {
        return torch::tensor(val, torch::dtype(torch::kFloat32));
    }
};

struct AddASTNode : public ASTNode {
    std::unique_ptr<ASTNode> left, right;
    AddASTNode(std::unique_ptr<ASTNode> l, std::unique_ptr<ASTNode> r) : left(std::move(l)), right(std::move(r)) {}
    at::Tensor eval(const std::unordered_map<std::string, at::Tensor>& env) override {
        return left->eval(env) + right->eval(env);
    }
};

struct SubASTNode : public ASTNode {
    std::unique_ptr<ASTNode> left, right;
    SubASTNode(std::unique_ptr<ASTNode> l, std::unique_ptr<ASTNode> r) : left(std::move(l)), right(std::move(r)) {}
    at::Tensor eval(const std::unordered_map<std::string, at::Tensor>& env) override {
        return left->eval(env) - right->eval(env);
    }
};

struct MulASTNode : public ASTNode {
    std::unique_ptr<ASTNode> left, right;
    MulASTNode(std::unique_ptr<ASTNode> l, std::unique_ptr<ASTNode> r) : left(std::move(l)), right(std::move(r)) {}
    at::Tensor eval(const std::unordered_map<std::string, at::Tensor>& env) override {
        return left->eval(env) * right->eval(env);
    }
};

struct DivASTNode : public ASTNode {
    std::unique_ptr<ASTNode> left, right;
    DivASTNode(std::unique_ptr<ASTNode> l, std::unique_ptr<ASTNode> r) : left(std::move(l)), right(std::move(r)) {}
    at::Tensor eval(const std::unordered_map<std::string, at::Tensor>& env) override {
        return left->eval(env) / right->eval(env);
    }
};

struct PowASTNode : public ASTNode {
    std::unique_ptr<ASTNode> left, right;
    PowASTNode(std::unique_ptr<ASTNode> l, std::unique_ptr<ASTNode> r) : left(std::move(l)), right(std::move(r)) {}
    at::Tensor eval(const std::unordered_map<std::string, at::Tensor>& env) override {
        return left->eval(env).pow(right->eval(env));
    }
};

class Parser {
    Lexer lexer;
    Token curr;
    void consume() { curr = lexer.next(); }
public:
    Parser(std::string src) : lexer(src) { consume(); }
    std::unique_ptr<ASTNode> parse() {
        return parse_expr();
    }
private:
    std::unique_ptr<ASTNode> parse_expr() {
        auto node = parse_term();
        while (curr.type == TOKEN_PLUS || curr.type == TOKEN_MINUS) {
            auto type = curr.type;
            consume();
            auto right = parse_term();
            if (type == TOKEN_PLUS) {
                node = std::make_unique<AddASTNode>(std::move(node), std::move(right));
            } else {
                node = std::make_unique<SubASTNode>(std::move(node), std::move(right));
            }
        }
        return node;
    }
    std::unique_ptr<ASTNode> parse_term() {
        auto node = parse_factor();
        while (curr.type == TOKEN_MUL || curr.type == TOKEN_DIV) {
            auto type = curr.type;
            consume();
            auto right = parse_factor();
            if (type == TOKEN_MUL) {
                node = std::make_unique<MulASTNode>(std::move(node), std::move(right));
            } else {
                node = std::make_unique<DivASTNode>(std::move(node), std::move(right));
            }
        }
        return node;
    }
    std::unique_ptr<ASTNode> parse_factor() {
        auto node = parse_primary();
        while (curr.type == TOKEN_POW) {
            consume();
            auto right = parse_primary();
            node = std::make_unique<PowASTNode>(std::move(node), std::move(right));
        }
        return node;
    }
    std::unique_ptr<ASTNode> parse_primary() {
        if (curr.type == TOKEN_NUMBER) {
            double v = std::stod(curr.text);
            consume();
            return std::make_unique<ConstNode>(v);
        }
        if (curr.type == TOKEN_IDENT) {
            std::string name = curr.text;
            consume();
            return std::make_unique<VarNode>(name);
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
};

class JITCompiledFunction {
    std::shared_ptr<ASTNode> root;
    std::string expr_str;
public:
    JITCompiledFunction(std::string expr) : expr_str(expr) {
        Parser parser(expr);
        root = parser.parse();
    }
    
    at::Tensor forward(const std::vector<at::Tensor>& inputs, const std::vector<std::string>& input_names) {
        std::unordered_map<std::string, at::Tensor> env;
        for (size_t i = 0; i < inputs.size(); ++i) {
            env[input_names[i]] = inputs[i];
        }
        return root->eval(env);
    }
    
    std::vector<at::Tensor> backward(
        const std::vector<at::Tensor>& inputs,
        const std::vector<std::string>& input_names,
        at::Tensor grad_output
    ) {
        std::unordered_map<std::string, at::Tensor> env;
        std::vector<at::Tensor> inputs_with_grad;
        for (size_t i = 0; i < inputs.size(); ++i) {
            auto var = torch::autograd::make_variable(inputs[i], true);
            env[input_names[i]] = var;
            inputs_with_grad.push_back(var);
        }
        auto out = root->eval(env);
        out.backward(grad_output);
        std::vector<at::Tensor> grads;
        for (auto& var : inputs_with_grad) {
            auto g = var.grad();
            if (g.defined()) {
                grads.push_back(g);
            } else {
                grads.push_back(torch::zeros_like(var));
            }
        }
        return grads;
    }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<JITCompiledFunction>(m, "JITCompiledFunction")
        .def(py::init<std::string>())
        .def("forward", &JITCompiledFunction::forward)
        .def("backward", &JITCompiledFunction::backward);
}
