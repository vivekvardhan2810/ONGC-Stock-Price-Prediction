# ONGC Stock Price Prediction using LSTM, SVM, and K-Nearest Neighbors (KNN)

This project demonstrates how to predict stock prices of ONGC using three machine learning models: LSTM, SVM, and K-Nearest Neighbors (KNN). Each model is evaluated based on its performance in predicting stock prices using various metrics, including RMSE (Root Mean Squared Error).

## Dataset

The dataset used for this project is the ONGC stock data, containing historical price information. The key feature used for prediction is the `Close` price.

## Models

The following models are implemented in this project:

1. **LSTM (Long Short-Term Memory)**: A type of Recurrent Neural Network (RNN) that is widely used for time series prediction.

2. **SVM (Support Vector Machine)**: A powerful machine learning algorithm that works well for both classification and regression tasks.

3. **K-Nearest Neighbors (KNN)**: A simple algorithm that stores all available cases and predicts the numerical target based on the closest K neighbors.

## Steps

### 1. Exploratory Data Analysis (EDA)
- Inspecting the dataset for missing values, trends, and summary statistics.
- Plotting the historical stock prices using a time-series line chart.

### 2. Data Preprocessing
- The `Close` price is used as the target for prediction.
- Data is normalized using the `MinMaxScaler` to scale values between 0 and 1.
- A time window (sequence) is created to prepare the data for time series prediction.

### 3. Splitting the Data
- The dataset is split into training and testing sets.
- A time step of 100 days is used to predict the next day's price.

### 4. Model Training and Evaluation
Each model is trained, evaluated, and its predictions are plotted:

#### 4.1 LSTM Model:
- Built using Keras Sequential API.
- Consists of two LSTM layers followed by a dense layer.
- Trained for 50 epochs with a batch size of 64.
- RMSE (Root Mean Squared Error) is used to evaluate the performance.

#### 4.2 SVM Model:
- Built using the `SVR` class from `sklearn.svm`.
- Trained using an RBF kernel.
- RMSE is used for performance evaluation.

#### 4.3 KNN Model:
- Built using the `KNeighborsRegressor` class from `sklearn.neighbors`.
- The number of neighbors (K) is set to 5.
- RMSE is used to evaluate the model.

### 5. Stock Price Prediction and Visualization
- The actual vs predicted stock prices are plotted for each model.
- Both training and testing results are visualized to compare model performance.

## Results
The performance of each model is evaluated using the RMSE metric. The smaller the RMSE value, the better the model's performance.

- **LSTM**:
  - Train RMSE: *value*
  - Test RMSE: *value*
  
- **SVM**:
  - Train RMSE: *value*
  - Test RMSE: *value*

- **KNN**:
  - Train RMSE: *value*
  - Test RMSE: *value*

## Dependencies

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Keras

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/your-username/stock-price-prediction.git
   cd stock-price-prediction
   ```

2. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Run the script:

   ```
   python stock_prediction.py
   ```

## File Structure

```
├── data/
│   └── ONGC.csv        # Dataset
├── models/
│   ├── lstm_model.py   # LSTM model
│   ├── svm_model.py    # SVM model
│   └── knn_model.py    # KNN model
├── README.md           # Project Documentation
└── stock_prediction.py # Main script to run the models
```

## Future Improvements

- Add more features (like ```Volume```, ```High```, ```Low``` prices) to improve prediction accuracy.

- Hyperparameter tuning for the SVM and KNN models.

- Implement other time series forecasting models such as ARIMA.

## License

This project is licensed under the MIT License.
