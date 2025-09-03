# Ensemble Learning and Random Forests 🌲

A comprehensive implementation of ensemble learning algorithms using scikit-learn.

## Overview

This repository demonstrates various ensemble learning techniques including Voting Classifiers, Bagging, Random Forest, and Boosting algorithms with practical examples and visualizations.

## Features

- **Voting Classifiers** (Hard & Soft voting)
- **Bagging** with Bootstrap Aggregating
- **Random Forest** with feature importance analysis
- **AdaBoost** and **Gradient Boosting**
- Model comparison and performance evaluation
- Hyperparameter tuning examples
- Comprehensive visualizations

## Quick Start

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load your data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate
accuracy = rf.score(X_test, y_test)
print(f"Accuracy: {accuracy:.3f}")
```

## Repository Structure

```
├── ensemble_learning_demo.ipynb    # Main notebook with all algorithms
├── models/
│   ├── voting_classifier.py       # Voting implementations
│   ├── bagging_random_forest.py   # Bagging and Random Forest
│   └── boosting_algorithms.py     # AdaBoost and Gradient Boosting
├── utils/
│   ├── data_preparation.py        # Data loading and preprocessing
│   └── visualization.py           # Plotting functions
└── README.md
```

## Algorithms Covered

| Algorithm | Best For | Key Parameters |
|-----------|----------|----------------|
| Voting Classifier | Combining different models | `voting='soft'` |
| Random Forest | General purpose, balanced performance | `n_estimators`, `max_features` |
| AdaBoost | Improving weak learners | `learning_rate`, `n_estimators` |
| Gradient Boosting | Maximum performance | `learning_rate`, `max_depth` |

## Requirements

```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

## Usage Examples

### Random Forest with Feature Importance
```python
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
importances = rf.feature_importances_
```

### Soft Voting Classifier
```python
voting_clf = VotingClassifier([
    ('rf', RandomForestClassifier()),
    ('svm', SVC(probability=True)),
    ('lr', LogisticRegression())
], voting='soft')
```

### Model Comparison
```python
models = {'Random Forest': RandomForestClassifier(), 
          'AdaBoost': AdaBoostClassifier()}
          
for name, model in models.items():
    score = cross_val_score(model, X, y, cv=5).mean()
    print(f"{name}: {score:.3f}")
```

## Results

The ensemble methods typically show 2-5% improvement over single algorithms:
- Single Decision Tree: ~85% accuracy
- Random Forest: ~90% accuracy  
- Gradient Boosting: ~92% accuracy

## Contributing

Feel free to open issues or submit pull requests to improve the implementations.
