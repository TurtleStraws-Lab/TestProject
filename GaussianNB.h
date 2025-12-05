#ifndef GAUSSIANNB_H
#define GAUSSIANNB_H

#include <vector>
#include <cmath>
#include <map>

struct GaussianNBModel {
    std::vector<int> classes;
    std::vector<std::vector<double>> means;      // means[class][feature]
    std::vector<std::vector<double>> variances;  // variances[class][feature]
    std::vector<double> priors;                  // prior probabilities per class
};

// Train Gaussian Naive Bayes
GaussianNBModel fit_gnb(const std::vector<std::vector<double>>& X,
                        const std::vector<int>& y);

// Predict labels
std::vector<int> predict_gnb(const GaussianNBModel& model,
                             const std::vector<std::vector<double>>& X);

// Compute macro F1 score
double macroF1_gnb(const std::vector<int>& y_true,
                   const std::vector<int>& y_pred);

#endif

