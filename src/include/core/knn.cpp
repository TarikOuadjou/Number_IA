#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <map>
#include <stdexcept>
#include "knn.h"

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

std::vector<int> findKNearestNeighbors(
    const std::vector<MNISTData>& data,  
    const std::vector<double>& input,    
    int k                               
) {
    std::vector<std::pair<double, int>> distances;

    for (size_t i = 0; i < data.size(); ++i) {
        double distance = euclideanDistance(input, data[i].pixels);
        distances.push_back({distance, static_cast<int>(i)});
    }

    std::sort(distances.begin(), distances.end());

    std::vector<int> neighbors;
    for (int i = 0; i < k && i < static_cast<int>(distances.size()); ++i) {
        neighbors.push_back(distances[i].second);
    }

    return neighbors;
}

int predictLabel(
    const std::vector<MNISTData>& data,  
    const std::vector<double>& input,    
    int k                                
) {
    std::vector<int> neighbors = findKNearestNeighbors(data, input, k);

    std::map<int, int> labelCounts;
    for (int neighbor : neighbors) {
        labelCounts[data[neighbor].label]++;
    }

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
