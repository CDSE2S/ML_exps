import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# %% Importing the dataset
df = pd.read_csv(r'D:\Studies\SEM 5\ai minor\Solar_radiation_classification.csv') 

# %% Viewing the dataset
print(df.head())
print("Data types:\n", df.dtypes)
print("Missing values:\n", df.isnull().sum())

# %% Handling categorical variables
# If you have categorical columns, convert them to numerical using one-hot encoding
df = pd.get_dummies(df, drop_first=True)

# %% Preparing the data
X = df.iloc[:, :-1].values  # Features (all columns except the last)
Y = df.iloc[:, -1].values    # Target variable (last column)

# %% Splitting the dataset into Training set and Test set
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.25, random_state=0)

# %% Fitting the Multiple Linear Regression model to the Training set
model = LinearRegression()
model.fit(X_Train, Y_Train)

# %% Making predictions on the Test set
Y_Pred = model.predict(X_Test)

# %% Evaluating the model
mse = mean_squared_error(Y_Test, Y_Pred)
r2 = r2_score(Y_Test, Y_Pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared: {r2:.2f}')

# %% Visualizing the results (for one feature if needed)
if X.shape[1] == 2:  # Check if there are exactly two features for a 2D plot
    plt.scatter(X_Test[:, 0], Y_Test, color='red', label='Actual')
    plt.scatter(X_Test[:, 0], Y_Pred, color='blue', label='Predicted')
    plt.title('Actual vs Predicted')
    plt.xlabel('Feature 1')  # Adjust accordingly
    plt.ylabel('Target')
    plt.legend()
    plt.show()