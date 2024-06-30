# importing libs
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# loading file
credit_card_data = pd.read_csv('/Users/anav_sobti/Downloads/creditcard.csv')

# distribution of legit and fraud (before balancing)
class_value_counts = credit_card_data['Class'].value_counts()
print("Class Distribution (Before Balancing):")
print(class_value_counts)

# data balancing
legit = credit_card_data[credit_card_data['Class'] == 0]
fraud = credit_card_data[credit_card_data['Class'] == 1]
legit_sample = legit.sample(495)  # Sample to match fraud count

new_df = pd.concat([fraud, legit_sample], axis=0, ignore_index=True)
print("\nClass Distribution (After Balancing):")
print(new_df['Class'].value_counts())

# splitting the data into features and target
X = new_df.drop(columns='Class', axis=1)
Y = new_df['Class']

# splitting data into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# model training (Logistic Regression)
model = LogisticRegression()

# training the logistic regression model with training data
model.fit(X_train, Y_train)

# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print("\nTraining Data Accuracy:", training_data_accuracy)

# accuracy on test data
X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print("Testing Data Accuracy:", testing_data_accuracy)

# Plot class distribution after balancing
plt.figure(figsize=(8, 6))
plt.bar(class_value_counts.index, class_value_counts.values)
plt.xlabel("Class")
plt.ylabel("Number of Transactions")
plt.title("Class Distribution After Balancing")
plt.show()
