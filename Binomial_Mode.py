import numpy as np
import pandas as pd

''' I will try to first apply the Binomial model first with
    for loops and after the same thing but using numpy arrays with vectorization,
    to see the difference in speed '''

'''
strike price = k,
underlying asset price = S,
risk-free rate = r,
time to maturity = T,
volatility = sigma,
number of steps = 
beta, value to find u and d
'''
''' shown at A Short Introduction to Computational Methods in Finance "Pedro R. S. Antunes" IST Técnico '''

def Binomial_Model(k, S, r, T, sigma, N):

    #  Constants
    dt = T / N  # Time step
    beta = 0.5 * ((1/np.exp(r*dt)) + (np.exp(r*dt) * np.exp(sigma**2 * dt))) # Value to find u and d
    u = beta + np.sqrt(beta ** 2 - 1) # Up Factor
    d = beta - np.sqrt(beta ** 2 - 1) # Down Factor
    p = (np.exp(r*dt)- d) / (u - d) # Risk-neutral probability
    print(f'Up Factor: {u}, Down Factor: {d}, Risk-neutral Probability: {p}')

    # Initialize stock prices at maturity
    S_t = np.zeros(N + 1)
    for j in range(N  + 1):
        S_t[j] = S * (u ** j) * d ** (N - j)

    S_t = np.maximum(k - S_t, 0) # Payoff at maturity put European Option

    # Backward induction for option price shown on Pag 76
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            S_t[j] = np.exp(-r * dt) * (p * S_t[j + 1] + (1 - p) * S_t[j])

    print(f'Option Price: {S_t[0]}')

Binomial_Model(k = 10, S = 5, r = 0.06, T = 1, sigma = 0.3, N = 20)


''' Now we will try to vectorize the code using numpy arrays '''