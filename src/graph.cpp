#include "graph.hpp"

Tensor* Graph::track(Tensor* t) {
    nodes.push_back(t);
    return t;
}

void Graph::clear() {
    for (Tensor* t : nodes) {
        delete t;
    }
    nodes.clear();
}

Graph::~Graph() {
    clear();
}