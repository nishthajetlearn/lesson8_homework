#Import the Required Libraries Start by importing the necessary libraries.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


#Get the Data into the Program Load the Titanic dataset into your program using pandas. Save the dataset as titanic.csv.
# Load the dataset
data = pd.read_csv('titanic.csv')

# Verify that the data has been successfully imported
print(data.head())
print(data.info())

#Data Preprocessing For this dataset, we'll need to preprocess the data. This includes converting categorical features into numerical values and handling missing values. We'll also define the target variable (Survived).
# Drop columns that won't be used for the model
data = data.drop(columns=['Name', 'Ticket', 'Cabin'], errors='ignore')

# Convert 'Sex' to numerical values: male = 0, female = 1
data["Sex"] = data["Sex"].replace({"male": 0, "female": 1})

# Fill missing values in the 'Age' column with the median age
data["Age"].fillna(data["Age"].median(), inplace=True)

# Fill missing values in 'Embarked' with the most common port
data["Embarked"].fillna(data["Embarked"].mode()[0], inplace=True)

# Convert 'Embarked' to numerical values: C = 0, Q = 1, S = 2
data["Embarked"] = data["Embarked"].replace({"C": 0, "Q": 1, "S": 2})

#Splitting the Data We will split the data into features (X) and labels (Y), using the Survived column as our target variable.

Y = data["Survived"]
X = data.drop("Survived", axis=1)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

#Model Selection and Training We’ll use the same Decision Tree Classifier as in the Iris dataset. Here's how to create and train the model:
model = DecisionTreeClassifier(max_depth=3, random_state=1)
model.fit(X_train, Y_train)

#Making Predictions After training, we can use our model to make predictions on the test set.
predictions = model.predict(X_test)

#Evaluating the Model Finally, we'll check the accuracy of our model:
print("Accuracy:", metrics.accuracy_score(predictions, Y_test))


