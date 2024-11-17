#ifndef KNN_H
#define KNN_H

#include <vector>

// Structure représentant une donnée MNIST
struct MNISTData {
    int label;                     // Le label (0-9)
    std::vector<double> pixels;    // Les 784 pixels de l'image, normalisés entre 0.0 et 1.0
};

// Prototype des fonctions
double euclideanDistance(const std::vector<double>& a, const std::vector<double>& b);
std::vector<int> findKNearestNeighbors(const std::vector<MNISTData>& data, const std::vector<double>& input, int k);
int predictLabel(const std::vector<MNISTData>& data, const std::vector<double>& input, int k);

#endif // KNN_H
