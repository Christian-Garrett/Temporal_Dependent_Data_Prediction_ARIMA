from pathlib import Path
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

module_path = Path(__file__).parents[1]
sys.path.append(str(module_path))


class DataPipeline:
    """
    A class used to create the data pipeline.
    ...

    Attributes
    ----------
    output_path : str
        Time series model output data text path
    data_folder : str
        Time series model input data text path
    time_series : df
        Time series input data
    arima_attribute_list : list (int)
        Parameter attributes for arima test models
    walk_list : list (int)
        Randomly generated values to represent a random walk
    arima_model_dict : dict
        2D dictionary containing test model transformations

    Methods
    -------
    preprocess_data()
        Load data, set index, change formats, visual sanity checks.
    perform_EDA()
        Check for stationarity, examine ARIMA attributes.
    train_models()
        Train 5 ARIMA models for comparison.
    evaluate_models()
        Compare models using AIC, LLR and visual analysis as well as
        extended model variations.

    """

    from MLPipeline.RandomWalk import build_random_walk
    from MLPipeline.WhiteNoise import generate_whitenoise
    from MLPipeline.Stationarity import check_stationarity
    from MLPipeline.Seasonality import check_seasonality
    from MLPipeline.WinterHolt import train_winterholt_model
    from MLPipeline.ARIMA import (train_ARIMA_models, 
                                  evaluate_arima_models,
                                  check_arima_variation_models)

    def __init__(self, data_path, output_path="Output/"):

        self.output_path=os.path.join(module_path, output_path)
        self.data_folder=data_path
        self.time_series=pd.read_excel(self.data_folder)
        self.arima_attribute_list = [(1,1,1), 
                                     (1,1,2), 
                                     (2,1,1), 
                                     (2,1,2),
                                     (1,2,1)]
        self.walk_list=None
        self.arima_model_dict=None

    def preprocess_data(self):

        self.time_series.set_index("month", inplace=True)
        self.time_series = self.time_series.asfreq('ME')
        self.time_series.Healthcare.plot(figsize=(20,5), title="Healthcare")
        plt.savefig(self.output_path+"healthcare.png")

    def perform_EDA(self):

        self.generate_whitenoise()
        self.build_random_walk()
        self.check_stationarity()
        self.check_seasonality()

    def train_models(self):

        self.train_winterholt_model()
        self.train_ARIMA_models()

    def evaluate_models(self):

        self.evaluate_arima_models()
        self.check_arima_variation_models()
