# Ex.No: 08     MOVINTG AVERAGE MODEL AND EXPONENTIAL SMOOTHING
### Date: 27-10-2025


### AIM:
To implement Moving Average Model and Exponential smoothing Using Python.
### ALGORITHM:
1. Import necessary libraries
2. Read the electricity time series data from a CSV file,Display the shape and the first 20 rows of
the dataset
3. Set the figure size for plots
4. Suppress warnings
5. Plot the first 50 values of the 'Value' column
6. Perform rolling average transformation with a window size of 5
7. Display the first 10 values of the rolling mean
8. Perform rolling average transformation with a window size of 10
9. Create a new figure for plotting,Plot the original data and fitted value
10. Show the plot
11. Also perform exponential smoothing and plot the graph
### PROGRAM:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Load data
data = pd.read_csv('/content/INDIA VIX_minute.csv')

# ✅ Make sure you have a datetime column (replace 'date' with your actual column name)
# Example: if your column is 'timestamp' or 'datetime', change it accordingly
data['date'] = pd.to_datetime(data['date'])  

# ✅ Set datetime column as index
data.set_index('date', inplace=True)

# Select 'open' column
passengers_data = data[['open']]
print("Shape of the dataset:", passengers_data.shape)
print("First 10 rows of the dataset:")
print(passengers_data.head(10))

# Plot original data
plt.figure(figsize=(12, 6))
plt.plot(passengers_data['open'], label='Original open Data')
plt.title('Original Data')
plt.xlabel('Date')
plt.ylabel('Open')
plt.legend()
plt.grid()
plt.show()

# Rolling mean
rolling_mean_5 = passengers_data['open'].rolling(window=5).mean()
rolling_mean_10 = passengers_data['open'].rolling(window=10).mean()

plt.figure(figsize=(12, 6))
plt.plot(passengers_data['open'], label='Original Data', color='blue')
plt.plot(rolling_mean_5, label='Moving Average (window=5)')
plt.plot(rolling_mean_10, label='Moving Average (window=10)')
plt.title('Moving Averages of Open')
plt.xlabel('Date')
plt.ylabel('Open')
plt.legend()
plt.grid()
plt.show()

data_monthly = data.resample('MS').sum()  # MS = Month Start
scaler = MinMaxScaler()
scaled_data = pd.Series(
    scaler.fit_transform(data_monthly['open'].values.reshape(-1, 1)).flatten(),
    index=data_monthly.index
)

print(scaled_data.head())


from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Assuming 'data' already has datetime index from previous step
data_monthly = data.resample('MS').sum()
data_monthly = data_monthly.asfreq('MS')  # Ensure monthly frequency

# Scaling and shifting to avoid non-positive values
scaler = MinMaxScaler()
scaled_data = pd.Series(
    scaler.fit_transform(data_monthly['open'].values.reshape(-1, 1)).flatten(),
    index=data_monthly.index
)
scaled_data = scaled_data + 1  # avoid zeros

# Train-test split
x = int(len(scaled_data) * 0.8)
train_data = scaled_data[:x]
test_data = scaled_data[x:]

# Model 1: Train and test visualization
model_add = ExponentialSmoothing(train_data, trend='add', seasonal='mul', seasonal_periods=12).fit()
test_predictions_add = model_add.forecast(steps=len(test_data))

# Plot train vs test vs forecast
ax = train_data.plot(figsize=(10,5))
test_data.plot(ax=ax)
test_predictions_add.plot(ax=ax)
ax.legend(["Train Data", "Test Data", "Forecast"])
ax.set_title('Holt-Winters Exponential Smoothing (Train/Test Split)')
plt.grid()
plt.show()

# RMSE Evaluation
rmse = np.sqrt(mean_squared_error(test_data, test_predictions_add))
print("Root Mean Squared Error (RMSE):", rmse)

# Variance and mean for reference
print("Scaled Data Variance:", np.sqrt(scaled_data.var()))
print("Scaled Data Mean:", scaled_data.mean())

# Model 2: Forecasting next year (12 months)
model_full = ExponentialSmoothing(scaled_data, trend='add', seasonal='mul', seasonal_periods=12).fit()
predictions = model_full.forecast(steps=12)

# Plot predictions
ax = scaled_data.plot(figsize=(10,5))
predictions.plot(ax=ax)
ax.legend(["Actual (Scaled)", "Forecast (Next Year)"])
ax.set_xlabel('Date')
ax.set_ylabel('Scaled Open Values')
ax.set_title('Forecasting Next Year using Exponential Smoothing')
plt.grid()
plt.show()

```
### OUTPUT:

<img width="750" height="298" alt="image" src="https://github.com/user-attachments/assets/e949ff22-f0f9-4586-b84d-fbede2bf741b" />

<img width="1124" height="613" alt="image" src="https://github.com/user-attachments/assets/a73a70a0-7a19-490a-a940-268aceaa96d3" />

Moving Average:

<img width="354" height="622" alt="image" src="https://github.com/user-attachments/assets/eb372be2-2662-4456-bb66-fe5885b96644" />

<img width="1120" height="607" alt="image" src="https://github.com/user-attachments/assets/d628f6e1-5fc4-4112-8a34-779d872abb30" />

Plot Transform Dataset:

<img width="627" height="60" alt="image" src="https://github.com/user-attachments/assets/90a7482e-a287-4525-8185-5f8cc536b6f9" />

Exponential Smoothing:

<img width="994" height="531" alt="image" src="https://github.com/user-attachments/assets/71131aa3-0078-4d83-893f-2c8421fb8e21" />

<img width="948" height="526" alt="image" src="https://github.com/user-attachments/assets/13d065c1-b447-463e-8941-8a0b7a30d270" />

### RESULT:
Thus we have successfully implemented the Moving Average Model and Exponential smoothing using python.
