import  pandas as pd
import matplotlib.pyplot as plt
from arima.code.modular.MLPipeline.Stationarity import Stationarity
from arima.code.modular.MLPipeline.RandomWalk import RandomWalk
from arima.code.modular.MLPipeline.WhiteNoise import WhiteNoise
from arima.code.modular.MLPipeline.Seasonality import Seasonality
from arima.code.modular.MLPipeline.WinterHolt import Winterholt
from arima.code.modular.MLPipeline.ARIMA import ARIMA_Model

# importing the data
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

ARIMA_Model().compute(df_comp)