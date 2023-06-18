# SDev.Python

Python repository for various tools and projects in Machine Learning for Quantitative Finance. In the current release,
we mostly work on stochastic volatility surfaces and their calibration through Machine Learning methods.

See other work on our main website [SDev-Finance](http://sdev-finance.com/).

## Stochastic volatility calibration

In this project we intend to use Neural Networks to improve the calibration speed for stochastic volatility models. For now
we consider only the direct map, i.e. the calculation from model parameters to implied volatilities.

We first generate datasets of parameters (inputs) and vanilla option prices (outputs) and then train the network to replicate the prices.
In this manner, the machine learning model is used as a pricing function to replace costly closed-forms or PDE/MC price calculations.

Our models can be saved to files for later usage, and can also be re-trained from a saved state. We cover (Hagan) SABR, Free-Boundary
SABR, ZABR and Heston models.


## Other Tools

The package contains various other tools including Black-Scholes/Bachelier formulas, Monte-Carlo simulation of vanilla prices and 
other utilities.