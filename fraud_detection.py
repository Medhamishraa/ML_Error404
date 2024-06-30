# importing libs
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#loading file
credit_card_data = pd.read_csv('/Users/anav_sobti/Downloads/creditcard.csv')
#print(credit_card_data.head())
#print(credit_card_data.tail())
#print(credit_card_data.isnull().sum())

#distribution of legit and fraud
#print(credit_card_data['Class'].value_counts())

legit = credit_card_data[credit_card_data['Class'] == 0]
fraud = credit_card_data[credit_card_data['Class'] == 1]
#print(legit['Amount'].describe())
#print(fraud['Amount'].describe())

grp_cc = credit_card_data.groupby('Class').count()
grp_cc['Count'] = grp_cc['Time']
#print(grp_cc)

legit_sample = legit.sample(495)

new_df = pd.concat([fraud , legit_sample] , axis=0, ignore_index=True)
print(new_df['Class'].value_counts())

#splitting the data into features and target
X = new_df.drop(columns='Class' , axis=1)
Y = new_df['Class']

# spliting data into training and testing

X_train , X_test , Y_train , Y_test = train_test_split(X , Y, test_size=0.2 , stratify=Y , random_state=2)

#model training (Logistic Regression)
model = LogisticRegression()

#training the logistic regression model with training data
model.fit(X_train , Y_train)

#accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction , Y_train)
print(training_data_accuracy)

#accuracy on test data
X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction , Y_test)
print(testing_data_accuracy)


