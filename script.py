import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate synthetic data
data = {
    'Reservoir_Pressure': np.random.uniform(2000, 5000, 100),
    'Temperature': np.random.uniform(100, 300, 100),
    'Historical_Output': np.random.uniform(100, 1000, 100),
    'Daily_Production': np.random.uniform(500, 1500, 100)
}

df = pd.DataFrame(data)
X = df[['Reservoir_Pressure', 'Temperature', 'Historical_Output']]
y = df['Daily_Production']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Linear Regression Results:")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
print(f"R² Score: {r2_score(y_test, y_pred)}")