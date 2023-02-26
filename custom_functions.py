import scipy.stats
import numpy as np
import matplotlib.pyplot as plt


def get_abs_err_confidence_interval(obs:np.ndarray, N:int=None, gg_mu:float=None, gg_sigma:float=None, alpha:float=0.05, sided:int=2, score:str="z") -> float:
    
    if gg_mu:
        mu = gg_mu
    else:
        mu = obs.mean()
    
    if gg_sigma:
        sigma = gg_sigma
    else:
        sigma = obs.std()

    n = len(obs)

    if N:
        endlichkeits_korrektur = (N-n)/(N-1)
    else:
        endlichkeits_korrektur = 1

    standard_fehler = (sigma / n ** 0.5) * endlichkeits_korrektur

    if score == "z":
        verteilungs_wert = scipy.stats.norm.ppf(q=1-(alpha/sided))
    elif score == "t":
        verteilungs_wert = scipy.stats.t.ppf(q=1-(alpha/sided), df=n-1)
    absoluter_fehler = verteilungs_wert * standard_fehler

    # t_intervall = mu - absoluter_fehler, mu + absoluter_fehler
    return absoluter_fehler

def get_sales_change_rates_percent(n:int) -> np.ndarray:
    observations = np.random.normal(loc=0, scale=0.2, size=n)
    return np.exp(observations) - 1


def show_data(x:np.ndarray, e:float=None) -> None:
    print(f"n={len(x)}, mean={x.mean()}, std={x.std()}")
    plt.hist(x, bins=50)
    plt.vlines(x.mean()+x.std(), ymin=0, ymax=len(x) / 10, colors="red", linestyles="dotted")
    plt.vlines(x.mean()-x.std(), ymin=0, ymax=len(x) / 10, colors="red", linestyles="dotted")
    plt.vlines(x.mean(), ymin=0, ymax=len(x) / 10, colors="red")
    if e:
        plt.vlines(x.mean()+e, ymin=0, ymax=len(x) / 10, colors="green", linestyles="dotted")
        plt.vlines(x.mean()-e, ymin=0, ymax=len(x) / 10, colors="green", linestyles="dotted")
    # plt.show()