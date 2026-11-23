import numpy as np
import pandas as pd
from scipy.stats import norm

from app.financial_command_center.options_risk import bs_call, implied_cdf, simulate

S, T, R, VOL = 100.0, 0.25, 0.04, 0.25


def flat_vol_chain():
    strikes = np.arange(70.0, 131.0, 5.0)
    call = bs_call(S, strikes, T, R, VOL)
    put = call - S + strikes * np.exp(-R * T)

    def quotes(mid):
        return pd.DataFrame({"strike": strikes, "bid": mid * 0.99, "ask": mid * 1.01, "volume": 100})

    return quotes(call), quotes(put)


def test_implied_cdf_recovers_lognormal_including_tails():
    growth, cdf = implied_cdf(*flat_vol_chain(), T, R)
    exact = norm.cdf((np.log(growth) - (R - VOL**2 / 2) * T) / (VOL * np.sqrt(T)))
    assert np.abs(cdf - exact).max() < 0.005
    assert growth[0] < 0.7 and growth[-1] > 1.3  # extends past the quoted strikes
    mean = np.sum(np.diff(cdf) * (growth[1:] + growth[:-1]) / 2)
    assert abs(mean - np.exp(R * T)) < 0.001  # risk-neutral: grows at the risk-free rate


def test_too_few_quotes_returns_none():
    calls, puts = flat_vol_chain()
    assert implied_cdf(calls.assign(volume=0), puts.assign(volume=0), T, R) is None


def test_correlation_widens_the_portfolio():
    m = implied_cdf(*flat_vol_chain(), T, R)
    spread = lambda rho: simulate([m, m], [100, 100], [[1, rho], [rho, 1]]).std()
    assert abs(spread(1.0) / spread(0.0) - np.sqrt(2)) < 0.02
