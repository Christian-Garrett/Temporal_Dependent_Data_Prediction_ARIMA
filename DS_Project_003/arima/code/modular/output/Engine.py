from pathlib import Path
import os
import sys
# import  pandas as pd
# import matplotlib.pyplot as plt
# from arima.code.modular.MLPipeline.Stationarity import Stationarity
# from arima.code.modular.MLPipeline.RandomWalk import RandomWalk
# from arima.code.modular.MLPipeline.WhiteNoise import WhiteNoise
# from arima.code.modular.MLPipeline.Seasonality import Seasonality
# from arima.code.modular.MLPipeline.WinterHolt import Winterholt
# from arima.code.modular.MLPipeline.ARIMA import ARIMA_Model

module_path = Path(__file__).parents[2]
sys.path.append(str(module_path))

from modular.MLPipeline import DataPipeline


# /Input/CallCenterData.xlsx
data_path = os.path.join(module_path, "input/Data-Chillers.csv")
dp_object = DataPipeline(data_path)

'''# importing the data
raw_csv_data = pd.read_excel("arima/code/modular/Input/CallCenterData.xlsx")

# check point of data
df_comp = raw_csv_data.copy()

df_comp.set_index("month", inplace=True)

# seeting the frequency as monthly
df_comp = df_comp.asfreq('M')


df_comp.Healthcare.plot(figsize=(20,5), title="Healthcare")
plt.savefig("arima/code/modular/Output/healthcare.png")

WhiteNoise().white_noise(df_comp)

RandomWalk().random_walk()

Stationarity().stationarity(df_comp)

Seasonality().seasonality(df_comp)

Winterholt().holt(df_comp)

ARIMA_Model().compute(df_comp)'''
