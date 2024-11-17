/* Include files*/
#include <SDL2/SDL.h>
#include <vector>   
#include <cmath>
#include <iostream>

class Layer
{
    public : 
        Layer(int numNodesIn,int numNodesOut);

        std::vector<double> Calculate(std::vector<double> input);

        int numNodesIn,numNodesOut;

        std::vector<std::vector<double>> weights;

        std::vector<double> biases; 
        
    private : 
};

class Neural
{   
    public :

        Neural(std::vector<int> LayerSizes);

        std::vector<double> CalculateOutputs(std::vector<double> input);

        void show_output(std::vector<double> input);

        double CalculateError(const std::vector<double>& targetOutput, const std::vector<double>& actualOutput);
        
        void afficher();

        void Train(const std::vector<std::vector<double>>& inputs,
               const std::vector<std::vector<double>>& targetOutputs,
               double learningRate, int epochs);

    private :
        std::vector<Layer> layers; 
};

