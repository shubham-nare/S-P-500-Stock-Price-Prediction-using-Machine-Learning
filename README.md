# S&P 500 Stock Price Prediction using Machine Learning

## Project Overview

This project aims to predict the direction of the S&P 500 (ticker `^GSPC`) stock price using historical data and a `RandomForestClassifier` machine learning model. The goal is to determine if the S&P 500's closing price will be higher or lower on the next trading day.

The project involves data acquisition, cleaning, feature engineering, model training, and backtesting to evaluate the model's performance over a significant historical period.

## Features

- **Data Acquisition:** Utilizes the `yfinance` library to download historical S&P 500 data.
- **Data Cleaning:** Removes unnecessary columns and handles missing values.
- **Target Variable Creation:** Defines the target variable (`Tomorrow`) indicating whether the next day's closing price is higher than the current day's.
- **Feature Engineering:** Creates new predictive features based on rolling averages and historical trends.
- **Machine Learning Model:** Implements a `RandomForestClassifier` for predicting stock price movement.
- **Backtesting System:** Develops a custom backtesting function to simulate trading over time and evaluate model performance.
- **Performance Metrics:** Uses precision score to assess the model's accuracy in predicting positive movements.
- **Visualization:** Plots the actual and predicted target values for a segment of the data.

## Technologies Used

- **Python:** The core programming language for the project.
- **`yfinance`:** For efficient stock data download from Yahoo Finance.
- **`pandas`:** For data manipulation and analysis, primarily with DataFrames.
- **`scikit-learn`:** For implementing the `RandomForestClassifier` and evaluating model performance (`precision_score`).
- **`matplotlib` (implicitly via pandas.plot):** For data visualization.

## Getting Started

### Prerequisites

Before running the notebook, ensure you have Python installed. Then, install the necessary libraries using pip:

```bash
pip install yfinance pandas scikit-learn matplotlib
```

### Installation and Usage

1. Clone the repository:

```bash
git clone https://github.com/shubham-nare/S-P-500-Stock-Price-Prediction-using-Machine-Learning.git
cd S-P-500-Stock-Price-Prediction-using-Machine-Learning
```

2. Open the Jupyter Notebook:

If you have Jupyter Notebook installed:

```bash
jupyter notebook Stock_Price_Prediction.ipynb
```

Alternatively, you can open the notebook directly in Google Colab by uploading the `.ipynb` file.

3. Run the cells:

Execute the cells in the notebook sequentially to download data, preprocess it, train the model, and view the predictions and performance metrics.

## Project Structure

- `Stock_Price_Prediction.ipynb`: The main Jupyter Notebook containing all the code for data acquisition, analysis, model building, and backtesting.
- `README.md`: This file, providing an overview of the project.

## Methodology

1. ### Data Download:

   - Historical S&P 500 data (`^GSPC`) is downloaded using `yfinance` with `period="max"` to get all available data.

2. ### Initial Data Inspection and Cleaning:

   - The `Date` column is set as the index and converted to datetime objects.
   - Irrelevant columns like Dividends and Stock Splits are removed.
   - The data is filtered to start from `'1990-01-01'` to focus on more recent and relevant market behavior.

3. ### Defining the Target:

   - A `Tomorrow` column is created by shifting the `Close` price by one day.
   - The `Target` column is then generated, which is `1` if `Tomorrow`'s closing price is greater than `Close` (indicating an upward movement), and `0` otherwise.

4. ### Feature Engineering:

   - **Rolling Averages:** For various horizons (2, 5, 60, 250, 1000 days), rolling means of the `Close` price are calculated.
   - **Close Ratios:** `Close_Ratio_X` features are created by dividing the current `Close` price by its rolling average over `X` days.
   - **Trend Indicators:** `Trend_X` features are calculated as the sum of the `Target` (daily upward movements) over the previous `X` days.
   - Missing values introduced by rolling calculations (`NaN`) are dropped.

5. ### Model Training and Backtesting:

   - A `RandomForestClassifier` is chosen due to its robustness and ability to handle various types of data.
   - A custom backtest function simulates the trading process:
     - It iterates through the data in chunks (e.g., training on the first 2500 days, then testing on the next 250, and so on).
     - For each iteration, the model is trained on the train set and makes predictions on the test set.
     - `predict_proba` is used to get the probability of an upward movement, and a threshold of 0.6 is applied to make a prediction (a more conservative approach for 'buy' signals).
   - Predictions are combined and evaluated.

## Evaluation

- The `precision_score` is used as the primary metric, focusing on the accuracy of positive predictions (when the model predicts the price will go up).
- `value_counts` for predictions and target are examined to understand the distribution of predictions and actual outcomes.

### Results

After backtesting the model with enhanced features, the following performance was observed:

```
Predictions
0.0    4586
1.0     870
Name: count, dtype: int64

Target
1    0.536937
0    0.463063
Name: count, dtype: float64

precision_score: 0.5712643678160919
```

The precision score indicates that when the model predicts an upward movement (`1.0`), it is correct approximately **57%** of the time. This is an improvement over random chance (which would be closer to the actual proportion of upward movements, around **53.7%**).

## Future Enhancements

- **More Advanced Features:** Explore other technical indicators (e.g., RSI, MACD, Bollinger Bands) or fundamental data.
- **Different Models:** Experiment with other machine learning algorithms like Gradient Boosting, SVMs, or neural networks (LSTMs).
- **Hyperparameter Tuning:** Optimize the `RandomForestClassifier`'s hyperparameters using techniques like `GridSearchCV` or `RandomizedSearchCV`.
- **Risk Management:** Incorporate strategies for position sizing and stop-loss orders.
- **Real-time Data:** Implement a system to fetch and predict with real-time data.
- **Volatility Indicators:** Add features that capture market volatility.
- **Sentiment Analysis:** Integrate sentiment data from news or social media.
