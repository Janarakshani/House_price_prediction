import pandas as pd
import numpy as np
import math
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Load the dataset
datagreen = pd.read_csv('housing-3.csv')

# Split features and target
datagreen_x = datagreen.drop(['RAD'], axis=1)
datagreen_y = datagreen['RAD']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(datagreen_x, datagreen_y, test_size=0.25, random_state=42)

# Train model
LR = LinearRegression()
LR.fit(x_train, y_train)

# Save model
with open('linear_model.pkl', 'wb') as f:
    pickle.dump(LR, f)

# Evaluation
y_predicted = LR.predict(x_test)
score = LR.score(x_test, y_test)
mse_lr_model = mean_squared_error(y_test, y_predicted)
rmse = math.sqrt(mse_lr_model)

print("Model Score (R²):", score)
print("MSE:", mse_lr_model)
print("RMSE:", rmse)
