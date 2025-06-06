## Developed By: Pooja A
## Reg No: 212222240072
## Date: 

# Ex.No: 07                                       AUTO REGRESSIVE MODEL



### AIM:
To Implementat an Auto Regressive Model using Python

### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.
5. Fit an AutoRegressive (AR) model with 13 lags
6. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
7. Make predictions using the AR model.Compare the predictions with the test data
8. Calculate Mean Squared Error (MSE).
9. Plot the test data and predictions.
   
### PROGRAM :
```python
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('seattle_weather.csv')

# Inspect the first few rows to understand the structure
print(data.head())

# Convert 'DATE' to datetime format and set it as the index
data['DATE'] = pd.to_datetime(data['DATE'])
data.set_index('DATE', inplace=True)

# Resample data to monthly frequency, calculating the mean precipitation per month
monthly_data = data['PRCP'].resample('M').mean().dropna()

# Check for stationarity using the Augmented Dickey-Fuller (ADF) test
result = adfuller(monthly_data)
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# Split the data into training and testing sets (80% training, 20% testing)
train_data = monthly_data.iloc[:int(0.8 * len(monthly_data))]
test_data = monthly_data.iloc[int(0.8 * len(monthly_data)):]

# Define the lag order for the AutoRegressive model based on ACF/PACF plots
lag_order = 12  # Monthly lag (seasonality) might be appropriate
model = AutoReg(train_data, lags=lag_order)
model_fit = model.fit()

# Plot Autocorrelation Function (ACF)
plt.figure(figsize=(10, 6))
plot_acf(monthly_data, lags=40, alpha=0.05)
plt.title('Autocorrelation Function (ACF) - Monthly Precipitation')
plt.show()

# Plot Partial Autocorrelation Function (PACF)
plt.figure(figsize=(10, 6))
plot_pacf(monthly_data, lags=40, alpha=0.05)
plt.title('Partial Autocorrelation Function (PACF) - Monthly Precipitation')
plt.show()

# Make predictions on the test set
predictions = model_fit.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1)

# Calculate Mean Squared Error (MSE) for predictions
mse = mean_squared_error(test_data, predictions)
print('Mean Squared Error (MSE):', mse)

# Plot Test Data vs Predictions
plt.figure(figsize=(12, 6))
plt.plot(test_data.index, test_data, label='Test Data - Monthly Precipitation', color='blue', linewidth=2)
plt.plot(test_data.index, predictions, label='Predictions - Monthly Precipitation', color='orange', linestyle='--', linewidth=2)
plt.xlabel('Date')
plt.ylabel('Precipitation (inches)')
plt.title('AR Model Predictions vs Test Data (Monthly Precipitation)')
plt.legend()
plt.grid(True)
plt.show()

```
### OUTPUT:

### GIVEN DATA:

![{B7194E27-ED49-4D14-B04B-917DAD200CA3}](https://github.com/user-attachments/assets/0c492183-aedd-4808-8b24-1762e3fc5c93)

### Augmented Dickey-Fuller test :

![{A59CC3E7-69A1-4D36-BFD7-DEA207744F3B}](https://github.com/user-attachments/assets/40741990-7fab-4e1d-a38a-1c315975d1af)


### PACF - ACF:

![{DDB1EA16-FC5C-4C73-9A88-96274F5534E4}](https://github.com/user-attachments/assets/15382a24-6e36-4db9-b76a-2f2feaa34d91)


![{B9AAF973-4D64-43F3-9588-2B5909F076E8}](https://github.com/user-attachments/assets/c73e833b-bf8f-4883-bb1d-7094245d7fb7)

### Mean Squared Error :

![{36D411A9-81B7-4A85-B101-F399C9B90727}](https://github.com/user-attachments/assets/7a04c1c8-4c70-431a-b7e3-8e34a343c9af)


### FINAL PREDICTION :
![{2F5292DA-1917-4360-B908-679D40D02EBE}](https://github.com/user-attachments/assets/02013918-6176-494e-a1a5-35fc1835b7e0)


### RESULT:
Thus, the auto regression function using python is successfully implemented.
