
Training:
I have trained the MealNoMeal dataset with Logistic Regression model.

dm_asgn2.py
Run the file to do the following:
1) Extract Features (DWT coeffient, Kurtosis, CONGA, LAGE, HGMI, LGMI)
2) Create normalized feature matrix
3) Apply Prinicipal Component Analysis(PCA) to get Top 5 components
4) k-fold cross validation(k=10)
5) Calculate beta values using Gradient Descent Function
6) Calculate Accuracy, Precision, Recall and F1 Score

OUTPUT:

Number of folds: 10

Overall Accuracy: 0.7030232558139535
Overall Recall: 0.6908565402306821
Overall Precision: 0.6995236642330463
Overall F1 Score: 0.6839839422526081

Testing:

test.py
Run the file to do the following:
1) Read beta values from beta.csv and test dataset.(Sample dataset: dataset1.csv)
2) Print Predicted labels 0: Nomeal and 1: Meal

Run command:
Give test dataset as argument while running the code
Example:
	python test.py dataset1.csv

OUTPUT:
Test sample no.: 1, Predicted Label: 0
Test sample no.: 2, Predicted Label: 0
Test sample no.: 3, Predicted Label: 1
Test sample no.: 4, Predicted Label: 0
Test sample no.: 5, Predicted Label: 0
Test sample no.: 6, Predicted Label: 1
Test sample no.: 7, Predicted Label: 1
...
Test sample no.: 430, Predicted Label: 1
Test sample no.: 431, Predicted Label: 0
Test sample no.: 432, Predicted Label: 0
Test sample no.: 433, Predicted Label: 0
Test sample no.: 434, Predicted Label: 0
Test sample no.: 435, Predicted Label: 1