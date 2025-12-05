#ifndef KNN_H
#define KNN_H

#include <vector>
#include <cstddef>

// KNN model struct
struct KNNModel {
    std::vector<std::vector<double>> X_train;
    std::vector<int> y_train;
    int k;
};

// Fit KNN (just store training data)
KNNModel fit_knn(const std::vector<std::vector<double>>& X,
                 const std::vector<int>& y,
                 int k = 5);

// Predict using KNN (Euclidean distance)
std::vector<int> predict_knn(const KNNModel& model,
                             const std::vector<std::vector<double>>& X_test);

// Compute macro F1 score for KNN predictions
double macroF1_knn(const std::vector<int>& y_true,
                   const std::vector<int>& y_pred);

#endif

