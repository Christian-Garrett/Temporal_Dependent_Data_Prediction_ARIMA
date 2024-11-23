import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
import statsmodels.graphics.tsaplots as sgt


def generate_whitenoise(self):

    wn = np.random.normal(loc=self.time_series.Healthcare.mean(), 
                          scale=self.time_series.Healthcare.std(), 
                          size=len(self.time_series))
    self.time_series["wn"] = wn
    self.time_series.wn.plot(figsize=(20, 5))
    plt.title("White noise time-series", size=24)
    plt.savefig(self.output_path+"whitenoise.png")
    autocorrelation_plot(self.time_series.wn)
    plt.savefig(self.output_path+"autocorr_wn.png")
    autocorrelation_plot(self.time_series.Healthcare)
    plt.savefig(self.output_path+"autocorr_health.png")
    sgt.plot_acf(self.time_series.wn, zero=False, lags=40)
    plt.title("ACF Of WN", size=20)
    plt.savefig(self.output_path+"act_wn.png")
    sgt.plot_acf(self.time_series.Healthcare, zero=False, lags=40)
    plt.title("ACF Of Healthcare", size=20)
    plt.savefig(self.output_path+"acf_health.png")
