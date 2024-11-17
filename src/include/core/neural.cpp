#include <iostream>
#include <cmath>
#include "neural.h"

double ActivationFunction(double weightedinput)
{
    return 1/(1+exp(weightedinput));
}

double Neural::CalculateError(const std::vector<double>& targetOutput, const std::vector<double>& actualOutput)
{
    double error = 0.0;
    for (size_t i = 0; i < targetOutput.size(); i++)
    {
        double diff = targetOutput[i] - actualOutput[i];
        error += diff * diff;
    }
    return error / targetOutput.size(); // Erreur quadratique moyenne (MSE)
}

Layer::Layer(int numNodesIn,int numNodesOut) : numNodesIn(numNodesIn), numNodesOut(numNodesOut)
{
    weights.resize(numNodesIn, std::vector<double>(numNodesOut));
    biases.resize(numNodesOut,0);   
}



std::vector<double> Layer::Calculate(std::vector<double> input)
{
    std::vector<double> calcul(numNodesOut,0);
    for(int i=0;i<numNodesOut;i++)
    {
        double somme = biases[i];
        for(int j=0;j<numNodesIn;j++)
        {
            somme += weights[j][i] * input[j];
        }
        calcul[i] = ActivationFunction(somme);
    }
    return calcul;
}

Neural::Neural(std::vector<int> LayerSizes)
{
    for(int i=0;i<LayerSizes.size()-1;i++)
    {
        layers.emplace_back(LayerSizes[i],LayerSizes[i+1]);
    }
}

std::vector<double> Neural::CalculateOutputs(std::vector<double> input)
{
    std::vector<double> input_bis(input);
    for(int i=0;i<layers.size();i++)
    {
        input_bis = layers[i].Calculate(input_bis);
    }
    return input_bis;
}

void Neural::show_output(std::vector<double> input)
{
    std::vector<double> output = CalculateOutputs(input);
    for(int i=0;i<output.size();i++)
    {
        std::cout<<output[i]<<std::endl;
    }
}

void Neural::afficher()
{
    for(int i=0;i<layers.size();i++)
    {
        std::cout<<layers[i].numNodesIn<<":"<<layers[i].numNodesOut<<std::endl;
    }
}