import numpy as np
from scipy . stats import truncnorm
def mutiplyGauss (m1 , s1 , m2 , s2):
    # computes the Gaussian distribution N(m,s) being propotional to N(m1 ,s1)*N(m2 ,s2)
    #
    # Input :
    # m1 , s1: mean and variance of first Gaussian
    # m2 , s2: mean and variance of second Gaussian
    #
    # Output :
    # m, s: mean and variance of the product Gaussian

    s = 1/(1/ s1 +1/ s2)
    m = (m1/s1+m2/s2)*s
    return m, s
def divideGauss (m1 , s1 , m2 , s2):
    # computes the Gaussian distribution N(m,s) being propotional to N(m1 ,s1)/N(m2 ,s2)
    #
    # Input :
    # m1 , s1: mean and variance of the numerator Gaussian
    # m2 , s2: mean and variance of the denominator Gaussian
    #
    # Output :
    # m, s: mean and variance of the quotient Gaussian

    m, s = mutiplyGauss (m1 , s1 , m2 , -s2)
    return m, s

def truncGaussMM (a, b, m0 , s0):
    # computes the mean and variance of a truncated Gaussian distribution
    #
    # Input :
    # a, b: The interval [a, b] on which the Gaussian is being truncated
    # m0 ,s0: mean and variance of the Gaussian which is to be truncated
    #
    # Output :
    # m, s: mean and variance of the truncated Gaussian
    # scale interval with mean and variance
    a_scaled , b_scaled = (a - m0) / np.sqrt(s0), (b - m0) / np.sqrt(s0)
    m = truncnorm .mean(a_scaled , b_scaled , loc=m0 , scale =np.sqrt(s0))
    s = truncnorm .var(a_scaled , b_scaled , loc=m0 , scale =np.sqrt(s0))
    return m, s

#messages are denoted as mu_#

#Initial variables
prior_s1_mean=25
prior_s2_mean=25
prior_s1_var=64
prior_s2_var=64
#factor functions 
factor_s1_mean=prior_s1_mean
factor_s2_mean=prior_s2_mean
factor_s2_var=16

mu_2_mean=prior_s1_mean
mu_2_var=prior_s2_mean
mu_3_mean=prior_s1_var
mu_3_var=prior_s2_var

mu_6_mean=mu_2_mean-mu_3_mean
mu_6_var=mu_2_var+mu_3_var+factor_s2_var

if y0 == 1:
    a, b = 0, np.Inf
else:
    a, b = np.NINF , 0