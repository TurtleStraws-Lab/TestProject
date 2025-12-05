#ifndef DECISIONTREE_H
#define DECISIONTREE_H

#include <vector>
#include <string>
#include <memory>
#include <map>

// Node structure for the tree
struct TreeNode {
    bool isLeaf;
    int label; // only valid if isLeaf
    int featureIndex; // index of feature to split
    double threshold; // for numeric features
    bool isNumeric; // true if numeric split
    std::map<std::string, std::shared_ptr<TreeNode>> children; // categorical splits
    std::shared_ptr<TreeNode> left;  // numeric <= threshold
    std::shared_ptr<TreeNode> right; // numeric > threshold

    TreeNode() : isLeaf(false), label(-1), featureIndex(-1), threshold(0.0), isNumeric(false) {}
};

// Model wrapper
struct DecisionTreeModel {
    std::shared_ptr<TreeNode> root;
};

// Function declarations
DecisionTreeModel fit_tree(const std::vector<std::vector<double>>& X, 
                           const std::vector<int>& y, int maxDepth = 10);

std::vector<int> predict_tree(const DecisionTreeModel& model, 
                              const std::vector<std::vector<double>>& X);

double computeAccuracy_tree(const std::vector<int>& y_true, const std::vector<int>& y_pred);

double macroF1_tree(const std::vector<int>& y_true, const std::vector<int>& y_pred);

#endif

