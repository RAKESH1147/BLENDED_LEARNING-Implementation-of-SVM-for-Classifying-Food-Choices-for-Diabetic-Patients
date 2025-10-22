# BLENDED LEARNING
# Implementation of Support Vector Machine for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a Support Vector Machine (SVM) model to classify food items and optimize hyperparameters for better accuracy.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load Data Import and prepare the dataset to initiate the analysis workflow.
   
2.Explore Data Examine the data to understand key patterns, distributions, and feature relationships.

3.Select Features Choose the most impactful features to improve model accuracy and reduce complexity.

4.Split Data Partition the dataset into training and testing sets for validation purposes.

5.Scale Features Normalize feature values to maintain consistent scales, ensuring stability during training.

6.Train Model with Hyperparameter Tuning Fit the model to the training data while adjusting hyperparameters to enhance performance.

7.Evaluate Model Assess the model’s accuracy and effectiveness on the testing set using performance metrics.

## Program:

/*
Program to implement SVM for food classification for diabetic patients.
Developed by: RAKESH K S
RegisterNumber: 212224040264
*/

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the dataset from the URL
data = pd.read_csv('food_items_binary.csv')

# Step 2: Data Exploration
# Display the first few rows and column names for verification
print(data.head())
print(data.columns)

# Step 3: Selecting Features and Target
# Define relevant features and target column
features = ['Calories', 'Total Fat', 'Saturated Fat', 'Sugars', 'Dietary Fiber', 'Protein']
target = 'class'  # Assuming 'class' is binary (suitable or not suitable for diabetic patients)

X = data[features]
y = data[target]

# Step 4: Splitting Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 6: Model Training with Hyperparameter Tuning using GridSearchCV
#Define the SVM model
svm = SVC()

# Set up hyperparameter grid for tuning
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

#Initialize GridSearchCV
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

#Extract the best model
best_model = grid_search.best_estimator_
print("Name: Rakesh.K.S")
print("Register Number: 212224040264")
print("Best Parameter:",grid_search.best_params_)

# Step 7: Model Evaluation
# Predicting on the test set
y_pred = best_model.predict(X_test)

# Calculate accuracy and print classification metrics
accuracy = accuracy_score(y_test, y_pred)
print("Name: Rakesh.K.S")
print("Register Number: 212224040264")
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


## Output:
![WhatsApp Image 2025-10-22 at 11 38 26_327fb8dc](https://github.com/user-attachments/assets/6c34f38b-b99c-4645-a7ef-d3d7a401011d)
![WhatsApp Image 2025-10-22 at 11 38 47_f099a7cf](https://github.com/user-attachments/assets/2c36c50c-9ad0-487e-bd28-736694781894)


## Result:
Thus, the SVM model was successfully implemented to classify food items for diabetic patients, with hyperparameter tuning optimizing the model's performance.
