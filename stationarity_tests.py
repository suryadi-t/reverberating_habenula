"""
This file computes the stationarity tests for a multistep regression (MR) procedure.
The multistep regression code can be found in https://github.com/jwilting/WiltingPriesemann2018
"""

import scipy.stats
import numpy as np
from scipy.optimize import curve_fit

def offset_func(k,b,m,c):
    return b*(m**k) + c

def linear(k,q1,q2):
    return q1*k + q2

def stationarity_test(b,m,k,r_k,fps):
    """
    Performs stationarity tests on the resulting MR estimation results to determine
    if the dataset is stationary, i.e. whether or not MR estimation is valid.
    
    b, m, k, r_k are all output from the MR estimation procedure, which fits
    the curve bm^k onto r_k vs k.
    
    Returns a string indicating the outcome of the stationarity test:
    "Clear" if dataset passes all tests, otherwise it will return the 
    name of the first test the dataset fails, in the following order:
    H_offset, H_tau. H_lin, MR_invalid, H_poisson

    Input
    ----------
    b : float
        Coefficient obtained from curve fitting r_k vs k using the equation bm^k.
        This is obtained from the MR estimation process.
    m : float
        The branching parameter obtained from the MR estimation process.
    k : 1d array of ints
        Number of lag timesteps.
    r_k : 1d array of floats
        Linear regression slope of the data against itself at lag k.
    fps : float
        Sampling rate of the dataset.
    """
    
    #compute residual from the MR estimation fit
    res = np.sum((b*(m**k)-r_k)**2)
    
    #Test 1: compare with residual from a fit with offset
    fit = curve_fit(offset_func,k,r_k,maxfev=10000)
    params = fit[0]
    res_offset = np.sum((offset_func(k,*params)-r_k)**2)

    if 2*res_offset < res: 
        return "H_offset"
    
    #Test 2: compare differences in autocorrelation time tau
    m_offset = fit[0][1] 
    delt = 1/fps
    tau = -delt/np.log(m)
    tau_offset = -delt/np.log(m_offset)
    if (np.abs(tau-tau_offset)/min(tau,tau_offset)) > 2:
        return "H_tau"
    
    #Test 3: compare residual against a linear model
    slope,intercept,r,pval,stderr = scipy.stats.linregress(k,r_k)
    res_linear = np.sum((linear(k,slope,intercept)-r_k)**2)
    if res_linear < res:
        return "H_lin"
        
    #Test 4: determine if mean of r_k is >0 using one-sided t-test
    t,p = scipy.stats.ttest_1samp(r_k, 0)
    if p/2 >= 0.1 or t<=0: #if fail to reject null in one-sided t-test
        if pval >= 0.05: #earlier linear fit indicates 0 slope
            return "H_poisson"
        else: #systematic trend exists
         return "MR_invalid" 
     
    return "Clear"