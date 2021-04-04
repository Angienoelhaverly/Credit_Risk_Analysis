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
Describe the balanced accuracy scores and the precision and recall scores of all six machine learning models based on three different criteria. 
1. Accuracy Score - a measure of how likely a model is to label all predictions correctly.
2. Preciscion - a classifier's ability to accurately label samples and minimize false positives or negatives
3. Recall (Sensitivity) - a classifier's ability to find all the positive or negative samples. In this scenario, the higher the recall, the less chance there is that a high risk applicant will be classified as low risk and vice versa.

### Naive Random Oversampling
The first model was trained with data sampled using the naive random overampling technique. In random oversampling, instances of the minority class are randomly selected and added to the training set until the majority and minority classes are balanced. Oversampling addresses class imbalance by duplicating or mimicking existing data.

![naive](https://user-images.githubusercontent.com/73972332/113496730-5b7d6980-94b1-11eb-96cb-7b9c99a26ee6.png)

### SMOTE Oversampling
The synthetic minority oversampling technique (SMOTE) is another oversampling approach to deal with unbalanced datasets. In SMOTE, like random oversampling, the size of the minority is increased. The key difference between the two lies in how the minority class is increased in size. In SMOTE, new instances are interpolated. That is, for an instance from the minority class, a number of its closest neighbors is chosen. Based on the values of these neighbors, new values are created. It's important to note that although SMOTE reduces the risk of oversampling, it does not always outperform random oversampling. Another deficiency of SMOTE is its vulnerability to outliers.

![smote](https://user-images.githubusercontent.com/73972332/113496731-5d472d00-94b1-11eb-9712-8106465f7c72.png)

### Cluster Centroids (Undersampling)
Undersampling is another technique to address class imbalance. Undersampling takes the opposite approach of oversampling. Instead of increasing the number of the minority class, the size of the majority class is decreased. Undersampling only uses actual data. On the other hand, undersampling involves loss of data from the majority class. Furthermore, undersampling is practical only when there is enough data in the training set. There must be enough usable data in the undersampled majority class for a model to be useful.

Cluster centroid undersampling is akin to SMOTE. The algorithm identifies clusters of the majority class, then generates synthetic data points, called centroids, that are representative of the clusters. The majority class is then undersampled down to the size of the minority class.

![undersampling](https://user-images.githubusercontent.com/73972332/113497339-fb3df600-94b7-11eb-9a68-9da261747c00.png)

### SMOTEENN (Combination Sampling)
In the next model, SMOTEENN is applied, instead of SMOTE. As with SMOTE, the minority class is oversampled; however, an undersampling step is added, removing some of each class's outliers from the dataset. The result is that the two classes are separated more cleanly. Resampling with SMOTEENN did not work miracles, but some of the metrics show an improvement over undersampling.

![smoteenn](https://user-images.githubusercontent.com/73972332/113496732-5e785a00-94b1-11eb-8733-574ef8cbcb58.png)

### Balanced Random Forest Classifier
We next tried two ensembles models, which improves overall model performance by combining multiple models to help improve accuracty and decrease variance. The Random Forests Classifier is composed of several small decision trees created from random sampling. By using the Balanced Random Forests, we oversample from the minority class to balance the classes.

![balanced random forest](https://user-images.githubusercontent.com/73972332/113496733-5fa98700-94b1-11eb-9d28-504c8249ab26.png)

### Easy Ensemble AdaBoost Classifier
The final model that we generated for this analysis was built using an easy ensemble classifier with adaptive boosting, or AdaBoost. Using AdaBoost, a model is trained and then evaluated. The errors of the first model are given extra weight when the subsequent model is trained and so on until the error rate is minimized.

![easy ensemble](https://user-images.githubusercontent.com/73972332/113496734-60dab400-94b1-11eb-92f0-59db83589841.png)

## Summary
While resampling can attempt to address imbalance, it does not guarantee better results.
