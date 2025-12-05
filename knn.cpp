#include "KNN.h"
#include <cmath>
#include <algorithm>

KNNModel fit_knn(const std::vector<std::vector<double>>& X_train, 
                 const std::vector<int>& y_train, int k) {
    KNNModel model;
    model.k = k;
    model.X_train = X_train;
    model.y_train = y_train;
    return model;
}

std::vector<int> predict_knn(const KNNModel& model, const std::vector<std::vector<double>>& X_test) {
    std::vector<int> y_pred;
    for (const auto& x : X_test) {
        // Compute distances
        std::vector<std::pair<double, int>> distances;
        for (size_t i = 0; i < model.X_train.size(); ++i) {
            double dist = 0.0;
            for (size_t j = 0; j < x.size(); ++j) {
                dist += (x[j] - model.X_train[i][j]) * (x[j] - model.X_train[i][j]);
            }
            distances.push_back({std::sqrt(dist), model.y_train[i]});
        }
        std::sort(distances.begin(), distances.end());
        // Majority vote among k nearest
        std::vector<int> votes;
        for (int i = 0; i < model.k; ++i) votes.push_back(distances[i].second);
        int count0 = std::count(votes.begin(), votes.end(), 0);
        int count1 = votes.size() - count0;
        y_pred.push_back((count1 > count0) ? 1 : 0);
    }
    return y_pred;
}

