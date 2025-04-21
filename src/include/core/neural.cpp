#include "neural.h"

std::vector<double> Softmax(const std::vector<double> &logits)
{
    std::vector<double> expValues(logits.size());
    double sumExp = 0.0;

    // Calcul des exponentielles de chaque valeur
    for (size_t i = 0; i < logits.size(); i++)
    {
        expValues[i] = exp(logits[i]);
        sumExp += expValues[i];
    }

    // Normalisation pour que la somme soit égale à 1
    for (size_t i = 0; i < logits.size(); i++)
    {
        expValues[i] /= sumExp;
    }

    return expValues;
}

double Neural::CalculateError(const std::vector<double> &targetOutput, const std::vector<double> &actualOutput)
{
    double error = 0.0;
    for (size_t i = 0; i < targetOutput.size(); i++)
    {
        // Appliquer la cross-entropy : -y * log(y_hat) pour chaque classe
        error -= targetOutput[i] * std::log(actualOutput[i] + 1e-15); // 1e-15 pour éviter log(0)
    }
    return error;
}

Layer::Layer(int numNodesIn, int numNodesOut, const std::string &activationFunctionName)
    : numNodesIn(numNodesIn), numNodesOut(numNodesOut), activationFunction(activationFunctionName)
{
    weights.resize(numNodesIn, std::vector<double>(numNodesOut));
    biases.resize(numNodesOut, 0);

    // Initialisation aléatoire des poids et des biais (exemple)
    for (int i = 0; i < numNodesIn; ++i)
    {
        for (int j = 0; j < numNodesOut; ++j)
        {
            weights[i][j] = ((double)rand() / RAND_MAX) * 2 - 1; // entre -1 et 1
        }
    }

    for (int i = 0; i < numNodesOut; ++i)
    {
        biases[i] = ((double)rand() / RAND_MAX);
    }
}

std::vector<double> Layer::Calculate(std::vector<double> input)
{
    std::vector<double> output(numNodesOut, 0);
    for (int i = 0; i < numNodesOut; i++)
    {
        double somme = biases[i];
        for (int j = 0; j < numNodesIn; j++)
        {
            somme += weights[j][i] * input[j];
        }
        output[i] = activationFunction.Function(somme);
    }
    return output;
}

Neural::Neural(std::vector<int> layerSizes, const std::string &activationFunctionName)
{
    for (size_t i = 0; i < layerSizes.size() - 1; i++)
    {
        layers.emplace_back(layerSizes[i], layerSizes[i + 1], activationFunctionName);
    }
}

std::vector<double> Neural::CalculateOutputs(std::vector<double> input)
{
    std::vector<double> input_bis(input);
    for (int i = 0; i < layers.size(); i++)
    {
        input_bis = layers[i].Calculate(input_bis);
    }
    return Softmax(input_bis); // Appliquer Softmax à la sortie finale
}

void Neural::show_output(std::vector<double> input)
{
    std::vector<double> output = CalculateOutputs(input);
    for (int i = 0; i < output.size(); i++)
    {
        std::cout << output[i] << std::endl;
    }
}

void Neural::afficher()
{
    for (int i = 0; i < layers.size(); i++)
    {
        std::cout << layers[i].numNodesIn << ":" << layers[i].numNodesOut << std::endl;
    }
}

void Neural::Train(const std::vector<std::vector<double>> &inputs,
                   const std::vector<std::vector<double>> &targetOutputs,
                   double learningRate, int epochs)
{
    std::cout << "Début de l'entraînement, nombre d'époques: " << epochs << std::endl;

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        std::cout << "Epoch " << epoch + 1 << " / " << epochs << std::endl;
        double totalError = 0.0;

        for (size_t i = 0; i < inputs.size(); i++)
        {
            const std::vector<double> &input = inputs[i];
            const std::vector<double> &target = targetOutputs[i];
            // === FORWARD PASS avec sauvegarde des activations ===
            std::vector<std::vector<double>> activations;
            std::vector<double> currentInput = input;
            activations.push_back(currentInput);

            for (Layer &layer : layers)
            {
                currentInput = layer.Calculate(currentInput);
                activations.push_back(currentInput);
            }

            std::vector<double> output = Softmax(activations.back());
            double error = CalculateError(target, output);
            totalError += error;
            // === BACKWARD PASS ===
            std::vector<std::vector<double>> layerDeltas(layers.size());
            std::vector<double> deltaOutput(output.size());
            for (size_t j = 0; j < output.size(); j++)
            {
                deltaOutput[j] = output[j] - target[j]; // y_hat - y
            }
            layerDeltas.back() = deltaOutput;

            // couches cachées
            for (int l = layers.size() - 2; l >= 0; l--)
            {
                const Layer &nextLayer = layers[l + 1];
                Layer &currentLayer = layers[l];
                const std::vector<double> &activation = activations[l + 1];

                std::vector<double> delta(currentLayer.numNodesOut, 0.0);

                for (int j = 0; j < currentLayer.numNodesOut; j++)
                {
                    double sumError = 0.0;
                    for (int k = 0; k < nextLayer.numNodesOut; k++)
                    {
                        sumError += nextLayer.weights[j][k] * layerDeltas[l + 1][k];
                    }
                    delta[j] = sumError * currentLayer.activationFunction.Derivative(activation[j]);
                }
                layerDeltas[l] = delta;
            }

            // changement des poids et biais ===
            for (size_t l = 0; l < layers.size(); l++)
            {
                Layer &layer = layers[l];
                const std::vector<double> &inputToLayer = activations[l];
                const std::vector<double> &delta = layerDeltas[l];

                for (int j = 0; j < layer.numNodesOut; j++)
                {
                    for (int k = 0; k < layer.numNodesIn; k++)
                    {
                        layer.weights[k][j] -= learningRate * delta[j] * inputToLayer[k];
                    }
                    layer.biases[j] -= learningRate * delta[j];
                }
            }
        }

        std::cout << "Epoch " << epoch + 1 << " / " << epochs << " - Erreur totale: " << totalError << std::endl;
        accuracy(inputs, targetOutputs); // Afficher la précision après chaque époque
        std::cout << "----------------------------------------" << std::endl;
    }
}

int Neural::Predict(const std::vector<double> &input)
{
    std::vector<double> probabilities = CalculateOutputs(input);
    int predictedClass = std::distance(probabilities.begin(), std::max_element(probabilities.begin(), probabilities.end()));
    return predictedClass;
}

void Neural::Export(const std::string &filename)
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Erreur d'ouverture du fichier " << filename << std::endl;
        return;
    }

    // Sauvegarder la structure des couches
    for (const Layer &layer : layers)
    {
        file << layer.numNodesIn << " " << layer.numNodesOut << std::endl;
    }

    // Sauvegarder les poids et les biais
    for (const Layer &layer : layers)
    {
        // Sauvegarder les poids
        for (int i = 0; i < layer.numNodesIn; i++)
        {
            for (int j = 0; j < layer.numNodesOut; j++)
            {
                file << layer.weights[i][j] << " ";
            }
        }
        file << std::endl;

        // Sauvegarder les biais
        for (int i = 0; i < layer.numNodesOut; i++)
        {
            file << layer.biases[i] << " ";
        }
        file << std::endl;
    }

    file.close();
    std::cout << "Réseau de neurones exporté avec succès dans " << filename << std::endl;
}

void Neural::Import(const std::string &filename, const std::string &activationFunctionName)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Erreur d'ouverture du fichier " << filename << std::endl;
        return;
    }

    layers.clear(); // Vider les couches actuelles du réseau

    // Lire la structure du réseau
    int numNodesIn, numNodesOut;
    while (file >> numNodesIn >> numNodesOut)
    {
        layers.emplace_back(numNodesIn, numNodesOut, activationFunctionName);
    }

    // Lire les poids et les biais
    for (Layer &layer : layers)
    {
        // Lire les poids
        for (int i = 0; i < layer.numNodesIn; i++)
        {
            for (int j = 0; j < layer.numNodesOut; j++)
            {
                file >> layer.weights[i][j];
            }
        }

        // Lire les biais
        for (int i = 0; i < layer.numNodesOut; i++)
        {
            file >> layer.biases[i];
        }
    }

    file.close();
    std::cout << "Réseau de neurones importé avec succès depuis " << filename << std::endl;
}

void Neural::accuracy(const std::vector<std::vector<double>> &inputs, const std::vector<std::vector<double>> &targetOutputs)
{
    std::vector<int> labels;
    for (const auto &target : targetOutputs)
    {
        int label = std::distance(target.begin(), std::max_element(target.begin(), target.end()));
        labels.push_back(label);
    }
    int correct = 0;
    for (size_t i = 0; i < inputs.size(); i++)
    {
        int predicted = Predict(inputs[i]);
        if (predicted == labels[i])
            correct++;
    }
    double accuracy = static_cast<double>(correct) / inputs.size() * 100.0;
    std::cout << "Précision : " << accuracy << "%" << std::endl;
}