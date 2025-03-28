# EndoCure

This repository - EndoCure is a predictive model for Females silently suffering from various issues which eventually leads to Endometriosis. 
This project is implemented in Python and Flask. The project aims to analyse how reading symptoms can help in early detection and prevention of Endometriosis in Women.

## Table of Contents
1. [Introduction](#introduction)
2. [Machine Learning Algorithms](#machine-learning-algorithms)
   - [Logistic Regression](#logistic-regression)
   - [Decision Trees](#decision-trees)
   - [Random Forest](#random-forest)
3. [Dataset](#dataset)
5. [Installation](#installation)
7. [License](#license)

## Introduction
Endometriosis is a condition in which tissue that normally lines the inside of the uterus begins to grow outside of it, in areas where it shouldn’t. This can lead to significant pain and discomfort, particularly during menstruation, and can impact daily life. Additionally, individuals with endometriosis may experience difficulties becoming pregnant, as the condition can cause scarring and blockages in the fallopian tubes. 
Currently, there is no specific therapy or cure for endometriosis, making early detection crucial. This project aims to use machine learning (ML) to develop a self-diagnostic tool that relies entirely on patient-reported symptoms.
The goal is to develop a user-friendly model for women in early medical evaluation that assesses the likelihood of endometriosis. The study identifies 24 key symptoms that predict the condition with 93% sensitivity and specificity. This method aims to reduce diagnosis time and highlight the most critical symptoms for accurate prediction.
## Dataset 

The dataset comprises 56 symptoms associated with endometriosis, compiled through a thorough review of relevant literature. It contains 800 entries, each describing specific symptoms related to endometriosis. This is not a continuous dataset; instead, each entry is labeled with a binary response (0 or 1), indicating whether the respective symptom is present or absent.

| Feature                  | Importance |
|--------------------------|------------|
| Painful Periods          | 0.536208   |
| Fatigue / Chronic fatigue| 0.054535   |
| Cysts (unspecified)      | 0.039391   |
| Lower back pain          | 0.037201   |
| Ovarian cysts            | 0.027674   |


## Machine Learning Algorithms
We applied several ML algorithms to train multiple endometriosis prediction models. Specifically, we applied Decision Trees, Random Forest and Logistic Regression. Besides generating predictions, these models also provide an importance analysis feature, which can be used to identify and remove non-contributing features from future surveys. Model performance was evaluated using common ML metrics : accuracy, sensitivity (recall), specificity, precision, F1-score,area under the ROC curve (AUC).

As discussed above, for each model type we also analyzed the effect of adding each symptom in the order of its importance based on the feature importance ranking derived from initial classification models (the models that were trained on the entire set of features)

### Logistic Regression

Logistic regression is a straightforward and interpretable algorithm, well-suited for binary classification tasks. It provides probability estimates, enabling easy interpretation of feature impacts. However, its effectiveness may diminish in the presence of non-linear relationships or a large number of features.
Below is the F1 Score and the AUC for the logistic regression model:

<img src="./images produced/Logistic Regression.png" alt="F1 Score" width="700"/>

*Figure 2: F1 Score and AUC*


| Metrics       | Mean    | Std        |
|---------------|---------|------------|
| Recall        | 0.9108  | 0.0437     |
| Specificity   | 0.9234  | 0.0591     |
| Precision     | 0.9318  | 0.0486     |
| F1-score      | 0.9196  | 0.0268     |
| Accuracy      | 0.9167  | 0.0291     |
| AUC           | 0.9171  | 0.0297     |

Accuracy achieved : 87.22%


### Decision Trees

This is a simple, tree-structured classifier, where internal nodes represent the features of a dataset, branches represent the decision rules, and each leaf node represents the outcome (class). The tree structure (organization of nodes) is determined based on the importance of the nodes using an attribute selection measure, such as information gain or Gini index. The model’s simplicity is both its weakness and its strength: On the one hand, this model is limited in its capacity to capture complex relationships between variables, yet on the other hand, its classification process is simple to interpret.
This is a graphical representation of how decision trees work : 


<img src="./images produced/decision_tree.png" alt="F1 Score" width="700"/>


Below is the F1 Score and the AUC for the decision trees:

<img src="./images produced/Decision Tree.png" alt="F1 Score" width="700"/>

*Figure 3: F1 Score and AUC*

| Metric      | Mean   | Std    |
|-------------|--------|--------|
| Recall      | 0.8919 | 0.0401 |
| Specificity | 0.8580 | 0.0672 |
| Precision   | 0.8763 | 0.0497 |
| F1-score    | 0.8827 | 0.0300 |
| Accuracy    | 0.8757 | 0.0339 |
| AUC         | 0.8750 | 0.0349 |

### Random Forest

This model generates a “forest” of decision trees, such that each tree is trained on a random subset of the features. The Random Forest model uses the entire collection of decision trees to classify a given sample, and eventually determines the classification output based on the trees’ majority vote, that is, the class that is the output of by most trees

<img src="./images produced/-random-forest.jpg" alt="F1 Score" width="700"/>

Below is the F1 Score and the AUC for the random forest:

<img src="./images produced/randomforest.png" alt="F1 Score" width="700"/>

*Figure 4: F1 Score and AUC*

Accuracy: 0.8764044943820225
Classification Report:
               precision    recall  f1-score   support

           0       0.84      0.90      0.87        80
           1       0.91      0.86      0.88        98

    accuracy                           0.88       178
   macro avg       0.88      0.88      0.88       178
weighted avg       0.88      0.88      0.88       178


## Results 

With the obtained results, we chose to utilize the Random Forest model as it demonstrated slightly superior performance. By assessing feature importance, we aimed to refine the model by focusing on the most relevant variables, enhancing its interpretability, and potentially further improving predictive accuracy. We plotted the ROC Curve and the Confusion Matrix and calculated the AUC for various numbers of features.

<img src="./images produced/ROC curve.jpeg" alt="F1 Score" width="700"/> 

*Figure 5: ROC curve*

<img src="./images produced/confusion matrix.jpeg" alt="F1 Score" width="700"/> 

*Figure 6: Confusion Matrix*

## Installation 

These instructions assume you have `git` installed for working with Github from command window.

1. Clone the repository, and navigate to the downloaded folder. Follow below commands.

```
git clone https://github.com/Manu04Tiwari/EndoCure.git
cd EndoCure
```
2. Install few required pip packages, which are specified in the requirements.txt file.

```
pip3 install -r requirements.txt
```

## License 
The code in this project is licensed under the MIT license 2025 - Manu Tiwari


