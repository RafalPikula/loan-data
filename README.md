# Loan Classification Project
The aim of this project is to explore the possibility of classifying the credit loans into two categories, namely, the good and the bad loans based on the LendingClub loan data. 

First, note that:
- the files containing the actual loan data are too big for upload but can be downloaded at this site: 
https://www.kaggle.com/husainsb/lendingclub-issued-loans
- the file LCDataDictionary.xlsx comes from this site: https://www.kaggle.com/wendykan/lending-club-loan-data

### Short Description
The whole process is divided into 4 Jupyter notebooks with pretty much self-explanatory names. The first two notebooks are devoted to the process of selecting, preparing and cleaning the data as well as removing the potential data leaks and selecting the suitable initial list of potential predictors. The third notebook contains the exploratory data analysis of the chosen and engineered features. Finally, the fourth notebook incorporates the **feature engineering** ideas presented during the exploratory analysis phase and includes the classification model training and selection (via the **cross-validation** procedure).

### More Detailed Description
Since the notebooks are well commented, we refrain from delving deeper into the minute details of the work done in the first three notebooks. The interested reader is encouraged to have a glance or two at their contents. We will, on the other hand, concentrate on the main part of the fourth notebook, pointing out the most important steps and observations:
- The data contains both numeric and categorical predictors that are dealt with separately in the preprocessing pipeline.
- There is a class imbalance: The positive class (bad loans, denoted by 1) is roughly one fourth of the negative class (good loans, denoted by 0).
- This imbalance makes the accuracy metric unsuitable for the classification problem. However, after much thought it was decided that neither the recall nor the f1 score were sufficient for the job. We would like to maximize the recall of both the positive and negative class, i.e., ideally, we would like to have many bad loans correctly classified as bad loans (to minimize loses) and simultaneously many good loans correctly classified as good loans (to maximize profits). Therefore, we employ a custom metric that is the product of these two recalls (which also happens to be the square of the geometric mean of these two quantities so, effectively, we are going to be maximizing this geometric mean).
- We select 4 classification methods, namely, the logistic regression (LR), linear support vector machine (SVM), random forest (RF) and gradient boosting (XGB) and perform the grid search cross-validation, selecting the option to balance the data via class weight parameter. Gradient boosting claims the best score.
- We also use the soft voting classifier (based on LR, RF and XGB since SVM does not support probability computations) in the hope of finding an even better classifier but to no avail. Its score, however, turns out to be pretty good.
- Finally, having considered the results, we choose our final model to be the weighted ensemble of these 5 classifiers.
