
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Example of loading time series data from a CSV file
df = pd.read_csv('https://raw.githubusercontent.com/saturn70/TS_Analysis/main/R/AirPassengers/AirPassengers.csv')
df.head()

import matplotlib.pyplot as plt

# Plot your time series data
plt.figure(figsize=(12, 6))
plt.plot(df['Month'],df['Passengers'])
plt.title('Time Series Data')
plt.xlabel('Month')
plt.ylabel('Passengers')
plt.show()

df['Month'] = pd.to_datetime(df['Month'])  # Convert to DateTime index
df.set_index('Month', inplace=True)

decomposition = sm.tsa.seasonal_decompose(df['Passengers'], model='additive')
decomposition

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
print(trend)
print(seasonal)
print(residual)

plt.figure(figsize=(12, 8))
plt.subplot(411)
plt.plot(df['Passengers'], label='Original')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(seasonal, label='Seasonal')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(residual, label='Residual')
plt.legend(loc='upper left')
plt.tight_layout()

from statsmodels.tsa.arima.model import ARIMA

# Define and fit an ARIMA model
model = ARIMA(df['Passengers'], order=(1, 1, 1))
results = model.fit()

# Print model summary
print(results.summary())

# Forecast future values
forecast_steps = 10  # Adjust the number of steps you want to forecast
forecast = results.forecast(steps=forecast_steps)

# Plot the original data and the forecast
plt.plot(df['Passengers'], label='Original')
plt.plot(range(len(df), len(df) + forecast_steps), forecast, label='Forecast', color='red')
plt.legend()
plt.show()

"""**ACF**"""

import statsmodels.api as sm
import matplotlib.pyplot as plt

# Assuming you have a time series in the 'data' variable
# Replace 'data' with your actual time series data

# Calculate the ACF
acf = sm.tsa.acf(df, fft=False)

# Create the ACF plot
plt.figure(figsize=(10, 6))
sm.graphics.tsa.plot_acf(df, lags=len(acf)-1)
plt.title('Autocorrelation Function (ACF) Plot')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.show()

"""**PACF**"""

import statsmodels.api as sm
import matplotlib.pyplot as plt

# Assuming you have a time series in the 'data' variable
# Replace 'data' with your actual time series data

# Calculate the PACF
pacf = sm.tsa.pacf(df)

# Create the PACF plot
plt.figure(figsize=(10, 6))
sm.graphics.tsa.plot_pacf(df, lags=len(pacf)-1)
plt.title('Partial Autocorrelation Function (PACF) Plot')
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation')
plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_error

# Calculate and print evaluation metrics
mse = mean_squared_error(df['Passengers'][-forecast_steps:], forecast)
mae = mean_absolute_error(df['Passengers'][-forecast_steps:], forecast)
print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
