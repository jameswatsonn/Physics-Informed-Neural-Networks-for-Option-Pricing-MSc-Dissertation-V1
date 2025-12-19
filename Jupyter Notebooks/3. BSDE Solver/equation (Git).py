import numpy as np
import tensorflow as tf


class Equation(object):
    """Base class for defining PDE related function."""

    def __init__(self, eqn_config):
        self.dim = eqn_config.dim
        self.total_time = eqn_config.total_time
        self.num_time_interval = eqn_config.num_time_interval
        self.delta_t = self.total_time / self.num_time_interval
        self.sqrt_delta_t = np.sqrt(self.delta_t)
        self.y_init = None

    def sample(self, num_sample):
        """Sample forward SDE."""
        raise NotImplementedError

    def f_tf(self, t, x, y, z):
        """Generator function in the PDE."""
        raise NotImplementedError

    def g_tf(self, t, x):
        """Terminal condition of the PDE."""
        raise NotImplementedError



class PricingDefaultRisk(Equation):
    """
    Nonlinear Black-Scholes equation with default risk in PNAS paper
    doi.org/10.1073/pnas.1718942115
    """
    def __init__(self, eqn_config):
        super(PricingDefaultRisk, self).__init__(eqn_config)
        self.x_init = np.ones(self.dim) * 100.0
        self.sigma = 0.2
        self.rate = 0.02   # interest rate R
        self.delta = 2.0 / 3
        self.gammah = 0.2
        self.gammal = 0.02
        self.mu_bar = 0.02
        self.vh = 50.0
        self.vl = 70.0
        self.slope = (self.gammah - self.gammal) / (self.vh - self.vl)

    def sample(self, num_sample):
        dw_sample = np.random.normal(size=[num_sample, self.dim, self.num_time_interval]) * self.sqrt_delta_t
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = (1 + self.mu_bar * self.delta_t) * x_sample[:, :, i] + (
                self.sigma * x_sample[:, :, i] * dw_sample[:, :, i])
        return dw_sample, x_sample

    def f_tf(self, t, x, y, z):
        piecewise_linear = tf.nn.relu(
            tf.nn.relu(y - self.vh) * self.slope + self.gammah - self.gammal) + self.gammal
        return (-(1 - self.delta) * piecewise_linear - self.rate) * y

    def g_tf(self, t, x):
        return tf.reduce_min(x, 1, keepdims=True)



class PricingDiffRate(Equation):
    """
    Nonlinear Black-Scholes equation with different interest rates for borrowing and lending
    in Section 4.4 of Comm. Math. Stat. paper doi.org/10.1007/s40304-017-0117-6
    """
    def __init__(self, eqn_config):
        super(PricingDiffRate, self).__init__(eqn_config)
        self.x_init = np.ones(self.dim) * 100
        self.sigma = 0.2
        self.mu_bar = 0.06
        self.rl = 0.04
        self.rb = 0.06
        self.alpha = 1.0 / self.dim

    def sample(self, num_sample):
        dw_sample = np.random.normal(size=[num_sample, self.dim, self.num_time_interval]) * self.sqrt_delta_t
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init
        factor = np.exp((self.mu_bar-(self.sigma**2)/2)*self.delta_t)
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = (factor * np.exp(self.sigma * dw_sample[:, :, i])) * x_sample[:, :, i]
        return dw_sample, x_sample

    def f_tf(self, t, x, y, z):
        temp = tf.reduce_sum(z, 1, keepdims=True) / self.sigma
        return -self.rl * y - (self.mu_bar - self.rl) * temp + (
            (self.rb - self.rl) * tf.maximum(temp - y, 0))

    def g_tf(self, t, x):
        temp = tf.reduce_max(x, 1, keepdims=True)
        return tf.maximum(temp - 120, 0) - 2 * tf.maximum(temp - 150, 0)





class BlackScholes(Equation):
    """1D European Call Option under Black-Scholes Model"""
    
    def __init__(self, eqn_config):
        super(BlackScholes, self).__init__(eqn_config)
        # Initialize parameters from configuration file 
        self.x_init = np.ones(self.dim) * 100.0  # Initial stock price
        self.strike = eqn_config.strike  # Strike
        self.rate = eqn_config.mu  # Risk-free rate 
        self.sigma = eqn_config.sigma   # Volatility
        self.dividend = getattr(eqn_config, 'dividend', 0.0)  # Dividend yield of 0
        
    def sample(self, num_sample):
        """Sample stock price paths using geometric Brownian motion"""
        # Generate Brownian increments
        dw_sample = np.random.normal(size=[num_sample, self.dim, self.num_time_interval]) * self.sqrt_delta_t
        
        # Initialize stock price tensor
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init
        
        # Exact solution for geometric Brownian motion
        drift = (self.rate - self.dividend - 0.5 * self.sigma**2) * self.delta_t
        for i in range(self.num_time_interval):
            diffusion = self.sigma * dw_sample[:, :, i]
            x_sample[:, :, i + 1] = x_sample[:, :, i] * np.exp(drift + diffusion)
            
        return dw_sample, x_sample

    def f_tf(self, t, x, y, z):
        return -self.rate * y  # -rV term from Black-Scholes PDE

    def g_tf(self, t, x):
        """Terminal condition (option payoff at expiration)"""
        # European call payoff: max(S-K, 0)
        return tf.nn.relu(x - self.strike)



class Heston(Equation):
    """Heston PDE for European call option
    
    PROPRIETARY CODE - CONFIDENTIAL
    Access to full code is restricted due to intellectual property considerations
    For code access requests or technical discussion, contact: jameswatson.business@outlook.com
    """


