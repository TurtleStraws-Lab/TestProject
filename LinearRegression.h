#ifndef LINEARREGRESSION_H
#define LINEARREGRESSION_H

#include <vector>

struct LinearModel {
    std::vector<double> weights;
    double bias;
};

// Closed-form linear regression (with optional L2 regularization)
LinearModel fit_linear(const std::vector<std::vector<double>>& X,
                       const std::vector<double>& y,
                       double lambda = 0.0);

// Predict
std::vector<double> predict_linear(const LinearModel& model,
                                   const std::vector<std::vector<double>>& X);

// Compute RMSE
double computeRMSE(const std::vector<double>& y_true,
                   const std::vector<double>& y_pred);

#endif

