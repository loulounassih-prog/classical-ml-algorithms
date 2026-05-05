# Classical Machine Learning Algorithms

This repository contains Python implementations of classical machine learning algorithms, written with a focus on understanding the underlying mechanics rather than relying on high-level model APIs.

The implementations cover nearest-neighbor methods, clustering, kernel methods, regression, classification, and support vector machines. Most components are built directly with NumPy and standard scientific Python tools.

## Overview

| Directory | Topic | Main components |
|---|---|---|
| `01-nearest-neighbor-search` | Nearest-neighbor search | Linear scan, KD-tree search |
| `02-k-means-clustering` | Clustering | K-means, centroid updates |
| `03-graph-clustering` | Graph-based clustering | Point clouds, graph construction, hierarchical clustering |
| `04-kernel-methods` | Kernel functions | Linear, polynomial, RBF, sigmoid and rational quadratic kernels |
| `05-knn-classification` | Classification | KNN classifier, confusion matrix, random projections |
| `06-regression-models` | Regression | KNN regression, linear regression |
| `07-kernel-svm` | Support Vector Machines | Soft-margin kernel SVM, dual optimization, grid-search cross-validation |

## Goals

The goal of this repository is to make the core ideas behind classical machine learning algorithms explicit:

- how data points are compared through distances or kernels;
- how clustering algorithms update assignments and centers;
- how classifiers produce predictions from learned parameters;
- how regression models estimate continuous targets;
- how model performance is evaluated;
- how hyperparameters can be selected through cross-validation.

This is not intended to be a production machine learning library. The focus is on clarity, mathematical structure, and implementation from first principles.

## Technical highlights

The repository includes implementations of:

- brute-force and tree-based nearest-neighbor search;
- Lloyd-style K-means clustering;
- graph-based clustering utilities;
- common kernel functions used in kernel methods;
- K-nearest-neighbor classification and regression;
- random projection utilities;
- linear regression;
- soft-margin kernel SVM trained through the dual quadratic program;
- grid-search cross-validation for SVM hyperparameter tuning.

## Dependencies

Install the required dependencies with:

```bash
pip install -r requirements.txt
```

Main dependencies:

```text
numpy
scipy
scikit-learn
qpsolvers[cvxopt]
```

## Repository notes

Datasets, grading scripts, cache files, and local experimentation scripts are intentionally excluded.

The repository is meant to provide clean educational implementations of the algorithms and their key computational steps.