import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as sgt
from  statsmodels.tsa.arima.model import ARIMA
from scipy.stats.distributions import chi2
from statsmodels.tsa.statespace.sarimax import SARIMAX


def check_arima_variation_models(self):
        
        # check exogenous model 
        arimaX_model_111 = ARIMA(self.time_series.Healthcare, 
                                      exog=self.time_series.Banking, 
                                      order=(1, 1, 1))
        arimaX_model_111_results = arimaX_model_111.fit()
        print(arimaX_model_111_results.summary())
        arimaX_model_111_residuals = arimaX_model_111_results.resid.iloc[:]

        sgt.plot_acf(arimaX_model_111_residuals, zero=False, lags=40)
        plt.title("ACF Of Residuals for ARIMAX(1,1,1)", size=20)
        plt.savefig(self.output_path+"acfx_111.png")

        # check seasonal model
        model_sarimax = SARIMAX(self.time_series.Healthcare, 
                                exog=self.time_series.Banking, 
                                order=(1, 1, 1), 
                                seasonal_order=(2, 0, 1, 5))
        s_arimaX_111_results = model_sarimax.fit()
        print(s_arimaX_111_results.summary())
        s_arimaX_111_residuals = s_arimaX_111_results.resid.iloc[:]

        sgt.plot_acf(s_arimaX_111_residuals, zero=False, lags=40)
        plt.title("ACF Of Residuals for SARIMAX(1,1,1)", size=20)
        plt.savefig(self.output_path+"acfsx_111.png")


def train_ARIMA_models(self):

    self.arima_model_dict = \
        {str(arima_attribute):
         {'model': ARIMA(self.time_series.Healthcare, order=arima_attribute),
          'report': ARIMA(self.time_series.Healthcare, order=arima_attribute).fit()}
         for arima_attribute in self.arima_attribute_list}


def LLR_test(L1, L2, DF=1):

    LR = (2 * (L2 - L1))
    p = chi2.sf(LR, DF).round(3)
    return p


def evaluate_arima_models(self):

    # AIC Tests
    for order_number, model_data in self.arima_model_dict.items():
        print(f"ARIMA{order_number}:  \t LL = {model_data['report'].llf},  \
              \t AIC = {model_data['report'].aic}")

    # LLR Tests
    print("\nLLR test p-value for order (1,1,1) vs (2,1,2) = " + \
          str(LLR_test(self.arima_model_dict['(1, 1, 1)']['report'].llf,
                       self.arima_model_dict['(2, 1, 2)']['report'].llf,
                       DF=2)))
    
    print("\nLLR test p-value for order (2,1,1) vs (2,1,2) = " + \
          str(LLR_test(self.arima_model_dict['(2, 1, 1)']['report'].llf, 
                       self.arima_model_dict['(2, 1, 2)']['report'].llf, 
                       DF=2)))
    
    # Plot residuals
    residuals_211 = self.arima_model_dict['(2, 1, 1)']['report'].resid
    sgt.plot_acf(residuals_211, zero=False, lags=40)
    plt.title("ACF Of Residuals for ARIMA(2,1,1)", size=20)
    plt.savefig(self.output_path+"acf_211.png")
