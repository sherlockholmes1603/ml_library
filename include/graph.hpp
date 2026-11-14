#ifndef GRAPH_HPP
#define GRAPH_HPP

#include <vector>
#include <Eigen/Dense>

class Tensor;

class Graph
{
public:
    std::vector<Tensor *> nodes; // All tensors in the graph

    Tensor *track(Tensor *t);
    void clear(); // Clear the graph (delete all tracked tensors)

    ~Graph(); // Destructor to clean up memory
};

#endif