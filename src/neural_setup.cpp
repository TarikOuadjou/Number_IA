#include <app/screen.h>
#include <core/neural.h>
#include <iostream>
#include <string>
#include <sstream>

std::vector<std::vector<double>> LoadMNISTData(const std::string &filename, std::vector<int> &labels)
{
    std::vector<std::vector<double>> inputs;
    std::ifstream file(filename);
    std::string line;

    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string value;
        std::vector<double> inputRow;

        // Lire le label (la première valeur de la ligne)
        std::getline(ss, value, ',');
        int label = std::stoi(value);
        labels.push_back(label);

        // Lire les pixels (les autres valeurs de la ligne)
        while (std::getline(ss, value, ','))
        {
            double pixel = std::stod(value) / 255.0; // Normaliser les pixels (valeurs entre 0 et 1)
            inputRow.push_back(pixel);
        }

        inputs.push_back(inputRow);
    }

    return inputs;
}

// Convertit un entier label en vecteur one-hot
std::vector<double> ToOneHot(int label, int numClasses = 10)
{
    std::vector<double> oneHot(numClasses, 0.0);
    oneHot[label] = 1.0;
    return oneHot;
}

int main(int argc, char *argv[])
{
    // Charger les données MNIST
    std::vector<int> labels;
    std::vector<std::vector<double>> inputs = LoadMNISTData("mnist_train.csv", labels);
    std::vector<std::vector<double>> targetOutputs;
    for (int label : labels)
        targetOutputs.push_back(ToOneHot(label));
    Neural network({784, 128, 64, 10}, "relu");
    network.Train(inputs, targetOutputs, 0.01, 5);
    network.Export("mnist_model_neural.txt");
    return 0;
}
