#include "tensor.hpp"
#include "graph.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <set>

Tensor::Tensor(int rows, int cols)
{
    data = Eigen::MatrixXf::Zero(rows, cols);
    grad = Eigen::MatrixXf::Zero(rows, cols);
}

Tensor::Tensor(int rows, int cols, std::string init_type)
{
    if (init_type == "zeros")
    {
        data = Eigen::MatrixXf::Zero(rows, cols);
    }
    else if (init_type == "ones")
    {
        data = Eigen::MatrixXf::Ones(rows, cols);
    }
    else if (init_type == "random")
    {
        data = Eigen::MatrixXf::Random(rows, cols);
    }
    else if (init_type == "xavier")
    {
        float limit = std::sqrt(6.0f / (rows + cols));
        data = Eigen::MatrixXf::Random(rows, cols) * limit;
    }
    else if (init_type == "he")
    {
        float limit = std::sqrt(2.0f / rows);
        data = Eigen::MatrixXf::Random(rows, cols) * limit;
    }
    else
    {
        throw std::invalid_argument("Unknown initialization type");
    }
    grad = Eigen::MatrixXf::Zero(rows, cols);
}

Tensor::Tensor(Eigen::MatrixXf m,
               std::vector<Tensor *> p,
               std::function<void()> bw,
               std::string operation)
    : data(m), parents(p), backward_fn(bw), op(operation)
{

    // Always initialize grad to zeros of the same shape as data
    grad = Eigen::MatrixXf::Zero(m.rows(), m.cols());
}

float &Tensor::operator()(int r, int c) { return data(r, c); }
float Tensor::operator()(int r, int c) const { return data(r, c); }

Tensor *Tensor::operator+(Tensor *other, Graph &graph)
{
    Eigen::MatrixXf res = this->data + other->data;

    // We "new" the result. Someone MUST "delete" this later.
    Tensor *out = new Tensor(res, {this, other}, nullptr, "add");

    out->backward_fn = [this, other, out]()
    {
        this->grad += out->grad;
        other->grad += out->grad;
    };

    return graph.track(out);
}

Tensor *Tensor::operator-(Tensor *other, Graph &graph)
{
    Eigen::MatrixXf res = this->data - other->data;

    // We "new" the result. Someone MUST "delete" this later.
    Tensor *out = new Tensor(res, {this, other}, nullptr, "sub");

    out->backward_fn = [this, other, out]()
    {
        this->grad += out->grad;
        other->grad -= out->grad; // Note the minus sign here for subtraction
    };
    return graph.track(out);
}

Tensor *Tensor::operator^(Tensor *other, Graph &graph)
{
    // .array() turns the matrix into an array to allow element-wise ops
    Eigen::MatrixXf res = this->data.array() * other->data.array();
    Tensor *out = new Tensor(res, {this, other}, nullptr, "hadamard");

    out->backward_fn = [this, other, out]()
    {
        // dL/dA = dL/dOut * B
        this->grad.array() += out->grad.array() * other->data.array();
        // dL/dB = dL/dOut * A
        other->grad.array() += out->grad.array() * this->data.array();
    };
    return graph.track(out);
}

Tensor *Tensor::operator*(Tensor *other, Graph &graph)
{
    Eigen::MatrixXf res = this->data * other->data;
    Tensor *out = new Tensor(res, {this, other}, nullptr, "dot");

    out->backward_fn = [this, other, out]()
    {
        this->grad += out->grad * other->data.transpose();
        other->grad += this->data.transpose() * out->grad;
    };
    return graph.track(out);
}

Tensor *Tensor::operator*(float scalar, Graph &graph)
{
    Eigen::MatrixXf res = this->data * scalar;
    Tensor *out = new Tensor(res, {this}, nullptr, "scale");

    out->backward_fn = [this, scalar, out]()
    {
        this->grad += out->grad * scalar;
    };
    return graph.track(out);
}

Tensor *Tensor::transpose(Graph &graph)
{
    Tensor *out = new Tensor(data.transpose(), {this}, nullptr, "transpose");
    out->backward_fn = [this, out]()
    {
        this->grad += out->grad.transpose();
    };
    return graph.track(out);
}

int Tensor::rows() const
{
    return data.rows();
}

int Tensor::cols() const
{
    return data.cols();
}

void Tensor::print() const
{
    std::cout << data << "\n";
}

float Tensor::sum() const
{
    return data.sum();
}

float Tensor::mean() const
{
    return data.mean();
}

Tensor *Tensor::sum(int axis)
{
    if (axis == 0)
    {
        return new Tensor(data.colwise().sum());
    }
    else if (axis == 1)
    {
        return new Tensor(data.rowwise().sum());
    }
    else
    {
        throw std::invalid_argument("Axis must be 0 or 1");
    }
}

Tensor *Tensor::relu()
{
    Eigen::MatrixXf res = data.cwiseMax(0.0f);
    Tensor *out = new Tensor(res, {this}, nullptr, "relu");

    out->backward_fn = [this, out]()
    {
        // Gradient is out->grad where data > 0, else 0
        Eigen::MatrixXf mask = (this->data.array() > 0).cast<float>();
        this->grad.array() += out->grad.array() * mask.array();
    };
    return out;
}

Tensor *Tensor::sigmoid()
{
    Eigen::MatrixXf res = 1.0f / (1.0f + (-data.array()).exp());
    Tensor *out = new Tensor(res, {this}, nullptr, "sigmoid");

    out->backward_fn = [this, out]()
    {
        // gradient = out * (1 - out) * upstream_grad
        this->grad.array() += out->data.array() * (1.0f - out->data.array()) * out->grad.array();
    };
    return out;
}

Tensor *Tensor::tanh()
{
    Eigen::MatrixXf res = data.array().tanh();
    Tensor *out = new Tensor(res, {this}, nullptr, "tanh");

    out->backward_fn = [this, out]()
    {
        // gradient = (1 - out^2) * upstream_grad
        this->grad.array() += (1.0f - out->data.array().square()) * out->grad.array();
    };
    return out;
}

Tensor *Tensor::softmax()
{
    // For numerical stability, subtract max from each row
    // Convert to array for element-wise operations
    Eigen::MatrixXf max_vals = data.rowwise().maxCoeff().replicate(1, data.cols());
    Eigen::MatrixXf stabilized = data - max_vals;
    Eigen::MatrixXf exp = stabilized.array().exp();
    Eigen::MatrixXf sum_exp = exp.rowwise().sum().replicate(1, data.cols());
    Eigen::MatrixXf result = exp.array() / sum_exp.array();
    Tensor *out = new Tensor(result, {this}, nullptr, "softmax");

    out->backward_fn = [this, out]()
    {
        // Softmax gradient is a Jacobian matrix, but we can simplify:
        // grad_input = out * (grad_output - sum(grad_output * out, axis=1))
        Eigen::MatrixXf dot_product = (out->grad.array() * out->data.array()).rowwise().sum();
        Eigen::MatrixXf grad_sum = dot_product.replicate(1, out->grad.cols());
        this->grad.array() += out->data.array() * (out->grad.array() - grad_sum.array());
    };
    return out;
}

Tensor *Tensor::mse_loss(Tensor *target)
{
    float loss_val = (this->data - target->data).array().square().mean();
    Tensor *out = new Tensor(Eigen::MatrixXf::Constant(1, 1, loss_val), {this, target}, nullptr, "mse");

    out->backward_fn = [this, target, out]()
    {
        // dMSE/dInput = 2/n * (Input - Target)
        float n = this->data.size();
        this->grad += (2.0f / n) * (this->data - target->data) * out->grad(0, 0);
    };
    return out;
}

float Tensor::MSE(const Tensor &other) const
{
    if (data.rows() != other.data.rows() || data.cols() != other.data.cols())
    {
        throw std::invalid_argument("Tensors must have the same shape for MSE");
    }
    return (data - other.data).array().square().mean();
}

float Tensor::CrossEntropy(const Tensor &other) const
{
    if (data.rows() != other.data.rows() || data.cols() != other.data.cols())
    {
        throw std::invalid_argument("Tensors must have the same shape for Cross-Entropy");
    }
    // Assuming data contains probabilities and other contains one-hot labels
    Eigen::MatrixXf clipped = data.array().max(1e-12).min(1 - 1e-12); // Avoid log(0)
    return -(other.data.array() * clipped.array().log()).sum() / data.rows();
}

Tensor *Tensor::cross_entropy_loss(Tensor *target)
{
    Eigen::MatrixXf clipped = data.array().max(1e-12f).min(1.0f - 1e-12f);
    float loss_val = -(target->data.array() * clipped.array().log()).sum() / data.rows();

    Tensor *out = new Tensor(Eigen::MatrixXf::Constant(1, 1, loss_val), {this, target}, nullptr, "cross_entropy");

    out->backward_fn = [this, target, out]()
    {
        float n = this->data.rows();
        // Gradient: -(target / data) * (1/n) * dL/dOut
        this->grad.array() += -(target->data.array() / this->data.array()) * (1.0f / n) * out->grad(0, 0);
    };
    return out;
}

void Tensor::zero_grad()
{
    grad.setZero();
}

void Tensor::backward()
{
    std::vector<Tensor *> topo;
    std::set<Tensor *> visited;

    std::function<void(Tensor *)> dfs = [&](Tensor *node)
    {
        if (visited.count(node))
            return;
        visited.insert(node);
        for (Tensor *parent : node->parents)
        {
            dfs(parent);
        }
        topo.push_back(node);
    };

    dfs(this);

    // Root gradient is 1.0 (dL/dL)
    this->grad = Eigen::MatrixXf::Ones(data.rows(), data.cols());

    // Iterate backwards (from loss to inputs)
    for (auto it = topo.rbegin(); it != topo.rend(); ++it)
    {
        if ((*it)->backward_fn)
        {
            (*it)->backward_fn();
        }
    }
}

Tensor *Tensor::softmax_cross_entropy_loss(Tensor *target)
{
    // https://levelup.gitconnected.com/killer-combo-softmax-and-cross-entropy-5907442f60ba

    // Step 1: Numerically stable softmax (subtract max per row)
    Eigen::MatrixXf max_vals = data.rowwise().maxCoeff().replicate(1, data.cols());
    Eigen::MatrixXf shifted = data - max_vals;
    Eigen::MatrixXf exp_val = shifted.array().exp();
    Eigen::MatrixXf row_sums = exp_val.rowwise().sum().replicate(1, data.cols());
    Eigen::MatrixXf probs = exp_val.array() / row_sums.array();

    // Step 2: Cross entropy loss
    // L = -sum(y * log(p)) / N
    Eigen::MatrixXf clipped_probs = probs.array().max(1e-12f);
    Eigen::MatrixXf log_probs = clipped_probs.array().log();
    float loss_val = -(target->data.array() * log_probs.array()).sum() / data.rows();

    Tensor *out = new Tensor(
        Eigen::MatrixXf::Constant(1, 1, loss_val),
        {this, target},
        nullptr,
        "softmax_cross_entropy");

    // ---- BACKWARD ----
    // Store probs in the lambda by value (critical — don't capture by ref,
    // probs is a local variable on the stack)
    out->backward_fn = [this, target, out, probs]()
    {
        // The magic gradient: (p - y) / N * upstream_grad
        float n = this->data.rows();
        this->grad.array() += (probs.array() - target->data.array()) / n * out->grad(0, 0);

        // target (y) gradient — usually not needed since labels are fixed,
        // but correct to implement
        // dL/dy = -log(p) / N
        target->grad.array() += -(probs.array().max(1e-12f).log() * out->grad(0, 0)) / n;
    };

    return out;
}

float Tensor::binary_cross_entropy(const Tensor &target) const
{
    Eigen::MatrixXf clipped = data.array().max(1e-12f).min(1.0f - 1e-12f);
    return -(target.data.array() * clipped.array().log() + (1.0f - target.data.array()) * (1 - clipped.array()).log()).mean();
}

Tensor *Tensor::binary_cross_entropy_loss(Tensor *target)
{
    Eigen::MatrixXf clipped = data.array().max(1e-12f).min(1.0f - 1e-12f);
    float loss_val = -(target->data.array() * clipped.array().log() + (1.0f - target->data.array()) * (1 - clipped.array()).log()).mean();

    Tensor *out = new Tensor(Eigen::MatrixXf::Constant(1, 1, loss_val), {this, target}, nullptr, "binary_cross_entropy");
    out->backward_fn = [this, target, out]()
    {
        float n = this->data.size();
        this->grad.array() += ((this->data.array() - target->data.array()) /
                               (this->grad.array() * (1 - this->data.array()))) *
                              out->grad(0, 0) / n;
    };
    return out;
}

Tensor *Tensor::mae_loss(Tensor *target)
{
    float loss_val = (this->data - target->data).array().abs().mean();
    Tensor *out = new Tensor(Eigen::MatrixXf::Constant(1, 1, loss_val), {this, target}, nullptr, "mae");

    out->backward_fn = [this, target, out]()
    {
        float n = this->data.size();
        // dMAE/dInput = sign(Input - Target) / n
        this->grad.array() += ((this->data.array() - target->data.array()).sign()) * out->grad(0, 0) / n;
    };
    return out;
}