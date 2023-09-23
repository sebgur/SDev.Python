# SDev.Python

Python repository for various tools and projects in Machine Learning for Quantitative Finance. In the current release,
we focus on stochastic volatility surfaces and their calibration through Machine Learning methods. See other work on our main 
website [SDev-Finance](http://sdev-finance.com/).

## Stochastic volatility calibration

In this project we use Neural Networks to improve the calibration speed for stochastic volatility models. For now
we consider only the direct map, i.e. the calculation from model parameters to implied volatilities.

We first generate datasets of parameters (inputs) and vanilla option prices (outputs) and then train the network to replicate the prices.
In this manner, the learning model is used as a pricing function to replace costly closed-forms or PDE/MC price calculations.

Our models can be saved to files for later usage, and can be re-trained from a saved state. We cover (Hagan) SABR, No-Arbitrage SABR
(i.e. the actual SABR dynamic), Free-Boundary SABR, ZABR and Heston models.

Jupyter notebooks are available for demo under ./notebooks.

Trained models are saved under ./models/stovol. Sample training data is provided under ./datasets/stovol. However, these are only small sets for demo
(50k samples). The larger sets we used for training (500k-2m) can be downloaded from our [Kaggle account](https://www.kaggle.com/sebastiengurrieri/datasets).

The notebook ./notebooks/StoVol Dataset Generation.ipynb can be used to generate samples (beware of setting up output paths to your local drive).
The notebook ./notebooks/StoVol Training.ipynb can be used to train models (pre-trained or not) on the samples. The .py scripts corresponding to these
notebooks are under sdevpy/projects/stovol.

## AAD Monte-Carlo

In script ./projects/aad/aad_mc.py, we show how to calculate 1st and 2nd order Greeks on a Monte-Carlo simulation of Black-Scholes model (1 asset) using AAD. We compare the results with standard Monte-Carlo Greeks by Finite Differences and the closed-form. We make use of payoff smoothers for both AAD and Standard MC.

In ./projects/aad/aad_mc_nd.py, we compare AAD to standard MC and Closed-Form on a product with generic dimension. We can then benchmark the performance of AAD compared to MC bump-based sensitivities varying dimension and number of simulations.

## Other Tools

The package contains various other tools including Black-Scholes/Bachelier formulas, Monte-Carlo simulation of vanilla prices and 
other utilities. It also features a wrapper class above Keras for easier management of trained models with their scalers,
as well as custom callbacks and learning schedules.

Jupyter notebooks of previous work are also available (PINNs, AAD Monte-Carlo) but are not yet integrated in the framework.