#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <map>
#include <stdexcept>
#include "knn.h"

// Fonction pour calculer la distance euclidienne entre deux points
double euclideanDistance(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Les vecteurs doivent avoir la même dimension");
    }

    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return std::sqrt(sum);
}

// Fonction pour trouver les k plus proches voisins
std::vector<int> findKNearestNeighbors(
    const std::vector<MNISTData>& data,  // Ensemble de données
    const std::vector<double>& input,    // Point de test
    int k                                // Nombre de voisins
) {
    // Vecteur pour stocker les distances et les indices
    std::vector<std::pair<double, int>> distances;

    // Calculer la distance entre le point d'entrée et tous les autres points
    for (size_t i = 0; i < data.size(); ++i) {
        double distance = euclideanDistance(input, data[i].pixels);
        distances.push_back({distance, static_cast<int>(i)});
    }

    // Trier par distance croissante
    std::sort(distances.begin(), distances.end());

    // Extraire les indices des k plus proches voisins
    std::vector<int> neighbors;
    for (int i = 0; i < k && i < static_cast<int>(distances.size()); ++i) {
        neighbors.push_back(distances[i].second);
    }

    return neighbors;
}

// Fonction pour prédire la classe (majorité des voisins)
int predictLabel(
    const std::vector<MNISTData>& data,  // Ensemble de données
    const std::vector<double>& input,    // Point de test
    int k                                // Nombre de voisins
) {
    // Trouver les k plus proches voisins
    std::vector<int> neighbors = findKNearestNeighbors(data, input, k);

    // Compter la fréquence des étiquettes des voisins
    std::map<int, int> labelCounts;
    for (int neighbor : neighbors) {
        labelCounts[data[neighbor].label]++;
    }

    // Trouver l'étiquette avec la fréquence maximale
    int predictedLabel = -1;
    int maxCount = 0;
    for (const auto& pair : labelCounts) {
        if (pair.second > maxCount) {
            maxCount = pair.second;
            predictedLabel = pair.first;
        }
    }

    return predictedLabel;
}
/*
int main() {
    // Exemple de données avec des pixels normalisés entre 0 et 1
    std::vector<MNISTData> dataset = {
        {0, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}}, // Données simplifiées (9 pixels au lieu de 784)
        {1, {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}},
        {0, {0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0}},
        {1, {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0}},
    };

    // Point d'entrée avec des pixels normalisés entre 0 et 1
    std::vector<double> input = {0.9, 0.9, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9}; // Point de test simplifié

    // Nombre de voisins
    int k = 3;

    // Prédire la classe du point d'entrée
    try {
        int predictedLabel = predictLabel(dataset, input, k);
        std::cout << "La classe prédite pour le point donné est : " << predictedLabel << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Erreur : " << e.what() << std::endl;
    }

    return 0;
}
*/