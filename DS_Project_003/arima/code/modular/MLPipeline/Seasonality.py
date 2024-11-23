import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose


def check_seasonality(self):

    # observed = Trend + Sesonal + Residual
    additive = \
        seasonal_decompose(self.time_series.Healthcare, 
                           model="additive")  # Naive decomposition Additive
    additive.plot()
    plt.savefig(self.output_path+"seasonal_additive.png")
    
    # observed = Trend * Sesonal * Residual
    additive = \
        seasonal_decompose(self.time_series.Healthcare, 
                           model="multiplicative")  # Naive decomposition Multiplicative
    additive.plot()
    plt.savefig(self.output_path+"seasonal_multiplicative.png")    
