# Diabetes-Classifier-Using-KNN
This project involves implementing a simple KNN classifier from scratch using the "diabetes.csv" dataset. The goal is to perform multiple iterations with different k values, utilizing Euclidean distance for distance computation. The dataset is divided into 70% for training and 30% for testing. Data preprocessing includes normalizing each feature using Log Transformation or Min-Max Scaling. Distance-Weighted Voting is employed to break ties in classifications.

## Tasks
Data Preprocessing  
* Normalize each feature column separately for training and test objects using Log Transformation or Min-Max Scaling.

KNN Classifier Implementation
KNN Algorithm:  
* Implement the KNN algorithm.
* Perform multiple iterations with different k values (e.g., K=2,3,4...).
* Use Euclidean distance for computing distances between instances.
* Implement Distance-Weighted Voting to break ties.
* Evaluation

For each iteration, output the value of k and the following summary information:  
Number of correctly classified instances.  
Total number of instances in the test set.  
Accuracy.  

Calculate and output the average accuracy across all iterations.
