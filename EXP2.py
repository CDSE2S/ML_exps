# solar_radiation_classification.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

# %% Importing the dataset
df = pd.read_csv(r'D:\Studies\SEM 5\ai minor\Solar_radiation_classification.csv')

# %% Viewing the dataset
print(df.head())

# Check for missing values
print(df.isnull().sum())

# %% Preparing the data
X = df.iloc[:, [2, 3]].values  # Features
Y = df.iloc[:, 4].values        # Target variable

# %% Splitting the dataset into Training set and Test set
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.25, random_state=0)

# %% Feature Scaling
sc_X = StandardScaler()
X_Train = sc_X.fit_transform(X_Train)
X_Test = sc_X.transform(X_Test)

# %% Fitting the Naive Bayes classifier into the Training set
classifier = GaussianNB()
classifier.fit(X_Train, Y_Train)

# %% Predicting the test set results
Y_Pred = classifier.predict(X_Test)
print(Y_Pred)

# %% Making the Confusion Matrix
cm = confusion_matrix(Y_Test, Y_Pred)
print(cm)

# %% Heatmap of Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(pd.DataFrame(cm), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# %% Calculating accuracy
accuracy = accuracy_score(Y_Test, Y_Pred)
print(f'Accuracy: {accuracy:.2f}')



