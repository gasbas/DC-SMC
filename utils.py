import numpy as np


def expnormalize_and_sum(log_proba) : 
    """Calculates the softmax function given the log weights. The denominator is also returned and will be used 
    to estimate log Z"""
    exp_pr = np.exp(log_proba - np.max(log_proba)) #maximum value is removed to avoid computation problems
    norm_pro, sum =  exp_pr / np.sum(exp_pr), np.max(log_proba) + np.log(np.sum(exp_pr))
    return norm_pro, sum 

def transform( p) : 
    """Simple logit function used to transform the probability p into natural point theta"""
    return np.log( p /(1-p))
    
def inverse_transform( logit) :
    """Simple logisitic function used to transform the natural point theta into probability p"""
    return 1 / (1 + np.exp(-logit))