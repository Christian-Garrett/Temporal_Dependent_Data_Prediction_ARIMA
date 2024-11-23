# ARIMA Time Series

Auto Regressive Integrated Moving Average (ARIMA) model is among one of the more popular and widely used statistical methods for time-series forecasting.
It is a class of statistical algorithms that captures the standard temporal dependencies that is unique to a time series data.
ARIMA is an acronym for “autoregressive integrated moving average.” It’s a model used in statistics and econometrics to measure events that happen over a period of time.
The model is used to understand past data or predict future data in a series. 
It’s used when a metric is recorded in regular intervals, from fractions of a second to daily, weekly or monthly periods. 

- Auto Regressive (AR) regression model is built on top of the autocorrelation concept, where the dependent variable depends on the past values of itself.
- Integrated(I): The integrated part of ARIMA attempts to convert the non-stationarity nature of the time-series data to something a little bit more stationary. By performing prediction on the difference between any two pair of observation rather than directly on the data itself.
- Moving Average(MA) attempts to reduce the noise in ourtime series data by performing some sort of aggregation operation to your past observations in terms of residual error.


## Autoregressor and MA Limitation

- It is assumed that both trend and seasonal components have been removed from your time series.
- This means that your time series is stationary, or does not show obvious trends (long-term increasing or decreasing movement) or seasonality (consistent periodic structure).
- Hence in order to model ARMA models we need to ensure stationarity for better predictions

## Time Series Basics

-   Chronological Data
- Cannot be shuffled
- Each row indicate specific time record
- Train – Test split happens chronologically
- Data is analyzed univariately (for given use case)
- Nature of the data represents if it can be predicted or not

## Code Description


    File Name : Engine.py
    File Description : Main class for starting different parts and processes of the lifecycle


    File Name : WhiteNoise.py
    File Description : Steps to test if the visualization is white noise or not


    File Name : RandomWalk.py
    File Description : Steps to test if the visualization is random walk or not


    File Name : Stationarity.py
    File Description : Steps to test if the visualization is stationary or not

    
    File Name : Seasonality.py
    File Description : Steps to test if the visualization is seasonal or not


    File Name : WinterHolt.py
    File Description : Holt-winters exponential smoothing code
    
    
    File Name : ARIMA.py
    File Description : ARIMA model code and evaluation


## Steps to Run

There are two ways to execute the end to end flow.

- Modular Code
- IPython

### Modular code

- Create virtualenv
- Install requirements `pip install -r requirements.txt`
- Run Code `python Engine.py`
- Check output for all the visualization

### IPython Google Colab

Follow the instructions in the notebook `ARIMA_.ipynb`

