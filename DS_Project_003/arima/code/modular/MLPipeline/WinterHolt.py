import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def train_winterholt_model(self):

    hw_model = ExponentialSmoothing(self.time_series.Healthcare.tolist())
    model_fit = hw_model.fit()
    # make prediction
    yhat = model_fit.predict(1, len(self.time_series))
    # check performance
    plt.figure(figsize=(20, 5))
    plt.plot(self.time_series.Healthcare.tolist())
    plt.plot(yhat.tolist(), color='red')
    plt.title("Holt Winter Model Prediction Vs Actual Healthcare")
    plt.legend(["actual", "predicted"])
    plt.savefig(self.output_path+"holtwinter.png") 
