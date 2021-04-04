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

* Balanced Accuracy Score: This model accurately predicts credit risk 62.5% of the time when the minority class is balanced by oversampling.
* Preciscion: The precision of this model is 0.01 for high risk and 1.00 for low risk applicants. This means that 100% of the predicted low risk applicants are actually low risk, but only 1% of the predicted high risk applicants are actually high risk.
* Recall: The recall of this model is 0.59 for high risk, and 0.67 for low risk applicants. This means that 59% of high risk applicants are classified as high risk and 67% of low risk applicants are classified as low risk.

![naive](https://user-images.githubusercontent.com/73972332/113496730-5b7d6980-94b1-11eb-96cb-7b9c99a26ee6.png)

### SMOTE Oversampling
The synthetic minority oversampling technique (SMOTE) is another oversampling approach to deal with unbalanced datasets. In SMOTE, like random oversampling, the size of the minority is increased. The key difference between the two lies in how the minority class is increased in size. In SMOTE, new instances are interpolated. That is, for an instance from the minority class, a number of its closest neighbors is chosen. Based on the values of these neighbors, new values are created. It's important to note that although SMOTE reduces the risk of oversampling, it does not always outperform random oversampling. Another deficiency of SMOTE is its vulnerability to outliers.
* Balanced Accuracy Score: The SMOTE oversampling model has a slightly lower accuracy score than Naive ROS. This model makes accurate predictions of credit risk 63% of the time.
* Preciscion: SMOTE oversampling gives the same model preciscion score as the model trained with Naive ROS (1.00 and 0.01 for low and high risk). Both models inaccurately classify 90% of high risk applicants as low risk.
* Recall: The recall for this model is slightly worse than the recall from the model trained with Naive ROS. 62% of high risk applicants are categorized as high risk and 64% of low risk applicants are classified as low risk.

![smote](https://user-images.githubusercontent.com/73972332/113496731-5d472d00-94b1-11eb-9712-8106465f7c72.png)

### Cluster Centroids (Undersampling)
Undersampling is another technique to address class imbalance. Undersampling takes the opposite approach of oversampling. Instead of increasing the number of the minority class, the size of the majority class is decreased. Undersampling only uses actual data. On the other hand, undersampling involves loss of data from the majority class. Furthermore, undersampling is practical only when there is enough data in the training set. There must be enough usable data in the undersampled majority class for a model to be useful.

Cluster centroid undersampling is akin to SMOTE. The algorithm identifies clusters of the majority class, then generates synthetic data points, called centroids, that are representative of the clusters. The majority class is then undersampled down to the size of the minority class.
* Balanced Accuracy Score: Undersampling the majority class gives the lowest accuracy score so far at 51.46%, which is only slightly better than the random naive method.
* Preciscion: The precision scores for this model are the same as the first two models.
* Recall: THe recall scores for this model are also the lowest thus far. Only 56% of high risk applicants and 47% of low risk applicants are classified correctly. If this model were used to predict risk and approve/deny accordingly, this model would classify more than half of low risk applicants as high risk and nearly half of high risk applicants as low risk.

![undersampling](https://user-images.githubusercontent.com/73972332/113497339-fb3df600-94b7-11eb-9a68-9da261747c00.png)

### SMOTEENN (Combination Sampling)
In the next model, SMOTEENN is applied, instead of SMOTE. As with SMOTE, the minority class is oversampled; however, an undersampling step is added, removing some of each class's outliers from the dataset. The result is that the two classes are separated more cleanly. Resampling with SMOTEENN did not work miracles, but some of the metrics show an improvement over undersampling.
* Balanced Accuracy Score: This model accurately predicts credit risk 63.75% of the time when the classes are balanced by combination over and undersampling.
* Preciscion: The precision scores for this model are the same as the first three models.
* Recall: This model correctly classifies 70% of high risk applicants and 57% of low risk applicants. This model has the best sensitivity for detecting high risk applicants out of all four sampling models.

![smoteenn](https://user-images.githubusercontent.com/73972332/113496732-5e785a00-94b1-11eb-8733-574ef8cbcb58.png)

### Balanced Random Forest Classifier
We next tried two ensembles models, which improves overall model performance by combining multiple models to help improve accuracty and decrease variance. The Random Forests Classifier is composed of several small decision trees created from random sampling. By using the Balanced Random Forests, we oversample from the minority class to balance the classes.
* Balanced Accuracy Score: This model accurately predicts credit risk 78.7% of the time when multiple models are combined and the minority class is balanced by oversampling.
* Preciscion: This model has the highest precision for classifying high risk applicants compared to models built from sampling techniques alone, but with a precision score of 0.04 this model still classifies 96% of high risk applicants as low risk. This model has the same preciscion score for classifying low risk applicants as the previous models (100%).
* Recall: This model correctly identifies 91% of low risk applicants as low risk, and 67% of high risk applicants as high risk. Though the recall score for high risk applicants is 3% lower than the recall score from the SMOTEENN model, the high recall score for low risk makes this ensemble model a better performer than models built from sampling techniques alone.

![balanced random forest](https://user-images.githubusercontent.com/73972332/113496733-5fa98700-94b1-11eb-9d28-504c8249ab26.png)

### Easy Ensemble AdaBoost Classifier
The final model that we generated for this analysis was built using an easy ensemble classifier with adaptive boosting, or AdaBoost. Using AdaBoost, a model is trained and then evaluated. The errors of the first model are given extra weight when the subsequent model is trained and so on until the error rate is minimized.
* Balanced Accuracy Score: This model accurately predicts credit risk 92.5% of the time when multiple models are trained sequentially on a balanced dataset to minimize error.
* Preciscion: The precision score for correclty identifying high risk applicants is 7%, which is the highest for all 6 models. The preciscion score for low risk applicants is 100%, which is the same as the other models.
* Recall: In this model, 91% of high risk and 94% of low risk applicants were correctly identified, which is the highest recall score of all the models.

![easy ensemble](https://user-images.githubusercontent.com/73972332/113496734-60dab400-94b1-11eb-92f0-59db83589841.png)

## Summary
While resampling can attempt to address imbalance, it does not guarantee better results. In terms of this dataset, we find that the best overall model to use is the "Easy Ensemble AdaBoost Classifier" because not only does it have a high accuracy score, but it also has the best precision and sensitivity (recall) especially in terms of correctly identifying high-risk applicants. Therefore, we would reccomend using this model to predict credit risk. 

The one downside to this model is a high false positive rate meaning that of the applicants that are predicted to be high risk, only a small amount of them will actually be high risk. In this scenario though, it is better for the model to have greater senstivity than precision because the credit card company would rather wrongly classify low risk applicants as high risk, than have high risk applicants classified as low risk thus approving them for a credit card or loan they are unable to repay. Also, the credit card company can further narrow down and examine these potential high risk individuals later on outside of the machine learning scope. 
