import numpy as np
import math
import scipy.stats as st

def blackScholes(s, x, t, r, v, divType = '', dividends = {}, prnt = True):
    """
    Function that values options using the Black-Scholes model
    ----------
    s : The current spot price of the underlying asset
    x : The strike price of the option contract
    t : The time to maturity of the option contract
    r : The risk-free rate over the life of the option
    v : The volatility of the underlying assets price
    
    Optional:
    divType : '' for no dividends (default), 'D' for discrete, 'C' for continuous
    dividends : {amount1: time1, amount2: time2} for discrete dividends, percentage for continuous dividend yield
    prnt : True to print option price to console (default), False to disable
    
    Returns
    -------
    Call Price, Put Price
    """
    #Discrete Dividends and No Dividends
    if divType in ['D', '']:
        d = 0
        for value in dividends:
            d += value * math.exp(-r * dividends[value])       
        d1 = (np.log((s - d)/ x) + (r + 0.5 * v ** 2) * t) / (v * math.sqrt(t))
        d2 = d1 - v * math.sqrt(t)
        nd1, nd2 = st.norm.cdf(d1), st.norm.cdf(d2)
        c = max((s - d) * nd1 - x * math.exp(-r * t) * nd2, 0)
        p = max(c - (s - d) + x * math.exp(-r * t), 0) #Put-Call Parity
    
    #Continuous Dividends
    elif divType == 'C':
        d1 = (np.log(s/ x) + (r - dividends + 0.5 * v ** 2) * t) / (v * math.sqrt(t))
        d2 = d1 - v * math.sqrt(t)
        nd1, nd2 = st.norm.cdf(d1), st.norm.cdf(d2)
        c = max((s) * math.exp(-dividends * t) * nd1 - x * math.exp(-r * t) * nd2, 0)
        p = max(c - (s) * math.exp(-dividends * t) + x * math.exp(-r * t), 0) #Put-Call Parity    
    
    if prnt:
        print(f"""The fair price of the call option is ${round(c, 4)}.
The fair price of the put option is ${round(p, 4)}""")
    return c, p

#Example Inputs
#No Dividends, $100 share price, $90 strike price, 9 month maturity
#8% risk free rate, 25% share price volatility
print('*** No Dividends ***')
blackScholes(100, 90, 0.75, 0.08, 0.25)

#Discrete Dividends, $1 dividend in month 1 and $1.50 dividend in month 7
print('\n*** Discrete Dividends ***')
blackScholes(100, 90, 0.75, 0.08, 0.25, 'D', {1: 1/12, 1.50: 7/12})

#Continuous Dividend Yield, 4% annually
print('\n*** Continuous Dividends ***')
blackScholes(100, 90, 0.75, 0.08, 0.25, 'C', 0.04)