#ifndef NEURAL_H
#define NEURAL_H

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>

// =================== ActivationFunction =================== //
struct ActivationFunction
{
    std::string name;

    ActivationFunction(const std::string &name = "sigmoid") : name(name) {}

    double Function(double x) const
    {
        if (name == "relu")
            return std::max(0.0, x);
        else if (name == "tanh")
            return std::tanh(x);
        else // sigmoid par défaut
            return 1.0 / (1.0 + std::exp(-x));
    }

    double Derivative(double x) const
    {
        if (name == "relu")
            return x > 0.0 ? 1.0 : 0.0;
        else if (name == "tanh")
            return 1.0 - std::tanh(x) * std::tanh(x);
        else // sigmoid
        {
            double sig = 1.0 / (1.0 + std::exp(-x));
            return sig * (1.0 - sig);
        }
    }
};

// =================== Layer =================== //
struct Layer
{
    int numNodesIn;
    int numNodesOut;
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
    ActivationFunction activationFunction;

    Layer(int numNodesIn, int numNodesOut, const std::string &activationFunctionName);

    std::vector<double> Calculate(std::vector<double> input);
};

// =================== Neural =================== //
class Neural
{
private:
    std::vector<Layer> layers;

public:
    Neural(std::vector<int> layerSizes, const std::string &activationFunctionName);

    std::vector<double> CalculateOutputs(std::vector<double> input);

    double CalculateError(const std::vector<double> &targetOutput,
                          const std::vector<double> &actualOutput);

    void Train(const std::vector<std::vector<double>> &inputs,
               const std::vector<std::vector<double>> &targetOutputs,
               double learningRate, int epochs);

    int Predict(const std::vector<double> &input);

    void Export(const std::string &filename);
    void Import(const std::string &filename, const std::string &activationFunctionName);

    void show_output(std::vector<double> input);
    void afficher();
    void accuracy(const std::vector<std::vector<double>> &inputs, const std::vector<std::vector<double>> &targetOutputs);
};

// =================== Utilitaire =================== //
std::vector<double> Softmax(const std::vector<double> &logits);

#endif // NEURAL_H
