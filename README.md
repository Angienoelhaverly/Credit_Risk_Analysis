# Credit_Risk_Analysis
## Project Overview
Apply machine learning to solve a real-world challenge: credit card risk.
### Project Task
Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. Credit card companies must evaluate new customer credit applications to assess the applicant's credit risk. 

The goal of this project is to build a classification model that can predict if an applicant is likelky to have low or high credit risk. The credit card company can use this information to determine whether or not an applicant should be approved. We will employ different techniques to train and evaluate models with unbalanced classes and evaluate models using resampling. We will then evaluate the performance of these models and make a written recommendation on whether they should be used to predict credit risk.

### Technical Deliverables
* Deliverable 1: Use Resampling Models to Predict Credit Risk
* Deliverable 2: Use the SMOTEENN Algorithm to Predict Credit Risk
* Deliverable 3: Use Ensemble Classifiers to Predict Credit Risk

## Results
Describe the balanced accuracy scores and the precision and recall scores of all six machine learning models.
### Naive Random Oversampling
The first model was trained with data sampled using the naive random overampling technique, which increases sampling from the minority class until both classes are balanced.
![naive](https://user-images.githubusercontent.com/73972332/113496730-5b7d6980-94b1-11eb-96cb-7b9c99a26ee6.png)

### SMOTE Oversampling
![smote](https://user-images.githubusercontent.com/73972332/113496731-5d472d00-94b1-11eb-9712-8106465f7c72.png)

### Cluster Centroids (Undersampling)
![undersampling](https://user-images.githubusercontent.com/73972332/113497339-fb3df600-94b7-11eb-9a68-9da261747c00.png)

### SMOTEENN (Combination Sampling)
![smoteenn](https://user-images.githubusercontent.com/73972332/113496732-5e785a00-94b1-11eb-8733-574ef8cbcb58.png)

### Balanced Random Forest Classifier
![balanced random forest](https://user-images.githubusercontent.com/73972332/113496733-5fa98700-94b1-11eb-9d28-504c8249ab26.png)

### Easy Ensemble AdaBoost Classifier
![easy ensemble](https://user-images.githubusercontent.com/73972332/113496734-60dab400-94b1-11eb-92f0-59db83589841.png)

## Summary
