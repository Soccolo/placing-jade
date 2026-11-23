"""Options-implied portfolio risk.

Each stock gets a risk-neutral distribution of its price at expiry, read off its
option chain (Breeden-Litzenberger). Stocks are then joined with a Gaussian copula
on their historical correlation and the portfolio is simulated.
"""
import numpy as np
import pandas as pd
from scipy.interpolate import splev, splrep
from scipy.optimize import brentq
from scipy.stats import norm


def bs_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d1 - sigma * np.sqrt(T))


def implied_vol(price, S, K, T, r):
    try:
        return brentq(lambda v: bs_call(S, K, T, r, v) - price, 1e-4, 5.0)
    except ValueError:  # price outside no-arbitrage bounds
        return np.nan


def liquid_mid(chain, min_volume, max_spread_ratio):
    """Keep tradable quotes and price them at the bid/ask mid."""
    spread_ratio = (chain["ask"] - chain["bid"]) / chain["ask"]
    chain = chain[(chain["volume"] >= min_volume) & (chain["bid"] > 0) & (spread_ratio <= max_spread_ratio)]
    return chain.assign(mid=(chain["bid"] + chain["ask"]) / 2)


# Assumed relative noise in a quoted implied volatility; sets how hard the smile is smoothed.
SMILE_NOISE = 0.02


def implied_cdf(calls, puts, T, r, min_volume=20, max_spread_ratio=0.2):
    """Risk-neutral CDF of a stock's growth factor (price at expiry / price now).

    Returns (growth, cdf), or None if there are too few liquid quotes.
    """
    calls = liquid_mid(calls, min_volume, max_spread_ratio)
    puts = liquid_mid(puts, min_volume, max_spread_ratio)
    both = calls.merge(puts, on="strike", suffixes=("_c", "_p"))
    if both.empty:
        return None
    # Spot implied by put-call parity at the money. Unlike the last traded price it is
    # in sync with the (delayed) option quotes and already net of dividends.
    near = both.loc[(both["mid_c"] - both["mid_p"]).abs().nsmallest(3).index]
    S = (near["strike"] * np.exp(-r * T) + near["mid_c"] - near["mid_p"]).median()

    # Out-of-the-money options only: the liquid side of each strike. Parity turns puts into call prices.
    calls, puts = calls[calls["strike"] >= S], puts[puts["strike"] < S]
    puts = puts.assign(mid=puts["mid"] + S - puts["strike"] * np.exp(-r * T))
    quotes = pd.concat([puts, calls]).sort_values("strike")
    quotes["iv"] = [implied_vol(p, S, k, T, r) for p, k in zip(quotes["mid"], quotes["strike"])]
    quotes = quotes.dropna(subset=["iv"])
    # One stale quote bends the whole smile: drop any that sits far from its neighbours.
    quotes = quotes[(quotes["iv"] / quotes["iv"].rolling(5, center=True, min_periods=1).median() - 1).abs() < 0.15]
    if len(quotes) < 4:
        return None

    strikes, ivs = quotes["strike"].to_numpy(), quotes["iv"].to_numpy()
    smile = splrep(strikes, ivs, w=1 / (SMILE_NOISE * ivs), s=len(strikes))
    lo, hi = strikes[0], strikes[-1]
    half_width = 6 * float(splev(S, smile)) * np.sqrt(T)
    prices = S * np.exp(np.linspace(-half_width, half_width, 2000))
    sigma = np.clip(splev(np.clip(prices, lo, hi), smile), 0.01, None)  # edge volatility held flat outside the quotes
    inside = (prices >= lo) & (prices <= hi)

    # Between the quoted strikes, Breeden-Litzenberger: P(price <= K) = 1 + e^{rT} dC/dK
    cdf = np.empty_like(prices)
    cdf[inside] = 1 + np.exp(r * T) * np.gradient(bs_call(S, prices, T, r, sigma)[inside], prices[inside])
    cdf[inside] = np.maximum.accumulate(np.clip(cdf[inside], 0, 1))
    # Beyond them the tails are kept, not cut off: each has the probability the quotes imply
    # at the last strike and the lognormal shape of that strike's volatility.
    lognormal = norm.cdf((np.log(prices / S) - (r - sigma**2 / 2) * T) / (sigma * np.sqrt(T)))
    i, j = np.flatnonzero(inside)[[0, -1]]
    cdf[:i] = cdf[i] * lognormal[:i] / lognormal[i]
    cdf[j + 1:] = 1 - (1 - cdf[j]) * (1 - lognormal[j + 1:]) / (1 - lognormal[j])
    return prices / S, cdf


def simulate(marginals, values, corr, n=100_000, seed=0):
    """Portfolio value at expiry, n samples. marginals: one (growth, cdf) per position; values: position sizes now."""
    # shortcut: Gaussian copula on historical correlation understates joint crashes, use a t-copula if tail risk drives decisions
    z = np.random.default_rng(seed).multivariate_normal(np.zeros(len(values)), corr, n)
    u = norm.cdf(z)
    return sum(v * np.interp(u[:, i], cdf, growth) for i, (v, (growth, cdf)) in enumerate(zip(values, marginals)))
