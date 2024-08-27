import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import pickle

# Step 1: Loading the dataset
url = "https://raw.githubusercontent.com/4GeeksAcademy/alternative-time-series-project/main/sales.csv"
df = pd.read_csv(url, parse_dates=['date'], index_col='date')

# Step 2: Construct and analyze the time series
plt.figure(figsize=(12,6))
plt.plot(df['sales'])
plt.title('Sales Time Series')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.show()

# Tensor analysis
print(f"Tensor of the time series: {df.index.freq}")

# Trend analysis
rolling_mean = df['sales'].rolling(window=12).mean()
rolling_std = df['sales'].rolling(window=12).std()
plt.figure(figsize=(12,6))
plt.plot(df['sales'], label='Original')
plt.plot(rolling_mean, label='Rolling Mean')
plt.plot(rolling_std, label='Rolling Std')
plt.legend()
plt.title('Sales, Rolling Mean & Standard Deviation')
plt.show()

# Stationarity test
result = adfuller(df['sales'].dropna())
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# Variability and noise
residuals = df['sales'] - rolling_mean
plt.figure(figsize=(12,6))
plt.plot(residuals)
plt.title('Residuals')
plt.show()

# Step 3: Train an ARIMA model
# For simplicity, we'll use a basic ARIMA(1,1,1) model
# In practice, you should use techniques like AIC or grid search to find optimal parameters

train = df[:int(0.8*len(df))]
test = df[int(0.8*len(df)):]

model = ARIMA(train['sales'], order=(1,1,1))
results = model.fit()

# Step 4: Predict with the test set
forecast = results.forecast(steps=len(test))
mse = mean_squared_error(test['sales'], forecast)
rmse = np.sqrt(mse)
print(f'Root Mean Squared Error: {rmse}')

plt.figure(figsize=(12,6))
plt.plot(train['sales'], label='Training Data')
plt.plot(test['sales'], label='Actual Test Data')
plt.plot(test.index, forecast, label='Forecast')
plt.legend()
plt.title('Sales Forecast vs Actual')
plt.show()

# Step 5: Save the model
with open('arima_model.pkl', 'wb') as f:
    pickle.dump(results, f)

print("Model saved as 'arima_model.pkl'")