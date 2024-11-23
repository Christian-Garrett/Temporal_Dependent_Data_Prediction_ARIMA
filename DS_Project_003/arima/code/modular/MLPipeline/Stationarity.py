import matplotlib.pyplot as plt
import scipy.stats
import pylab
import statsmodels.tsa.stattools as sts


def check_stationarity(self):

    sts.adfuller(self.time_series.wn)  # Dickey fuller stationarity test
    sts.adfuller(self.time_series.Healthcare)
    scipy.stats.probplot(self.time_series.wn, plot=pylab)  # The QQ plot for gausian test
    plt.title("QQ plot for White Noise")
    pylab.savefig(self.output_path+"qq_wn.png")
    scipy.stats.probplot(self.time_series.Healthcare, plot=pylab)
    plt.title("QQ plot for Healthcare")
    pylab.savefig(self.output_path+"qq_healthcare.png")   
