#include "optimizer.hpp"

// ---------- Optimizer (base) ----------

Optimizer::Optimizer(std::vector<Tensor*> parameters, float learning_rate) {
    params = parameters;
    lr = learning_rate;
}

void Optimizer::zero_grad() {
    for (Tensor* t : params) {
        t->zero_grad();
    }
}

// ---------- SGD with Momentum ----------

SGD::SGD(std::vector<Tensor*> p, float learning_rate, float m) : Optimizer(p, learning_rate), momentum(m) {
    for (Tensor* t : params) {
        velocity.push_back(Eigen::MatrixXf::Zero(t->rows(), t->cols()));
    }
}

void SGD::step() {
    for(size_t i=0; i< params.size(); i++){
        velocity[i] = momentum * velocity[i] - lr * params[i]->grad;
        params[i]->data += velocity[i];
    }
}

// ---------- Adam ----------

Adam::Adam(std::vector<Tensor*> p, float learning_rate, float b1, float b2, float eps) 
        : Optimizer(p, learning_rate), beta1(b1), beta2(b2), epsilon(eps) {
    for (Tensor* t : params) {
        m.push_back(Eigen::MatrixXf::Zero(t->rows(), t->cols()));
        v.push_back(Eigen::MatrixXf::Zero(t->rows(), t->cols()));
    }
}

void Adam::step() {
    t++;
    for(size_t i=0; i< params.size(); i++){
        m[i] = beta1 * m[i] + (1 - beta1) * params[i]->grad;
        v[i] = beta2 * v[i] + (1 - beta2) * params[i]->grad.cwiseProduct(params[i]->grad);

        Eigen::MatrixXf m_hat = m[i] / (1 - std::pow(beta1, t));
        Eigen::MatrixXf v_hat = v[i] / (1 - std::pow(beta2, t));

        params[i]->data -= lr * m_hat.cwiseQuotient((v_hat.cwiseSqrt() + Eigen::MatrixXf::Constant(v_hat.rows(), v_hat.cols(), epsilon)));
    }
}

// ---------- RMSProp ----------

RMSProp::RMSProp(std::vector<Tensor*> p, float learning_rate, float alpha, float eps, float m) 
        : Optimizer(p, learning_rate), alpha(alpha), epsilon(eps), momentum(m) {
    for (Tensor* t : params) {
        cache.push_back(Eigen::MatrixXf::Zero(t->rows(), t->cols()));
        velocity.push_back(Eigen::MatrixXf::Zero(t->rows(), t->cols()));
    }
}

void RMSProp::step() {
    for(size_t i=0; i< params.size(); i++){
        cache[i] = alpha * cache[i] + (1-alpha) * params[i]->grad.cwiseProduct(params[i]->grad);

        Eigen::MatrixXf update = params[i]->grad.cwiseQuotient(
            cache[i].cwiseSqrt() + Eigen::MatrixXf::Constant(cache[i].rows(), cache[i].cols(), epsilon)
        );

        if(momentum > 0.0f) {
            velocity[i] = momentum*velocity[i]-lr*update;
            params[i]->data += velocity[i];
        }
        else{
            params[i]->data -= lr*update;
        }
    }
}

// ---------- Adagrad ----------

Adagrad::Adagrad(std::vector<Tensor*> p, float learning_rate, float eps) 
        : Optimizer(p, learning_rate), epsilon(eps) {
    for (Tensor* t : params) {
        cache.push_back(Eigen::MatrixXf::Zero(t->rows(), t->cols()));
    }
}

void Adagrad::step() {
    for(size_t i=0;i<params.size();i++){
        cache[i] += params[i]->grad.cwiseProduct(params[i]->grad);
        params[i]->data -= lr * params[i]->grad.cwiseQuotient(
            cache[i].cwiseSqrt() + Eigen::MatrixXf::Constant(cache[i].rows(), cache[i].cols(), epsilon)
        );
    }
}

// ---------- SGD without Momentum ----------

SGD_NoMomentum::SGD_NoMomentum(std::vector<Tensor*> p, float learning_rate) : Optimizer(p, learning_rate) {}

void SGD_NoMomentum::step() {
    for(size_t i=0; i< params.size(); i++){
        params[i]->data -= lr * params[i]->grad;
    }
}

