# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 17:08:44 2020

@author: HBK
"""
from skopt import gp_minimize
from skopt.learning import GaussianProcessRegressor
# from skopt.learning.gaussian_process.kernels import ConstantKernel, RBF
from skopt.learning.gaussian_process.kernels import ConstantKernel, Matern
# from skopt.plots import plot_convergence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skopt.sampler import Lhs
from skopt.space import Space
from skopt import dump, load
from QTransport_cls import OpenQT, NCohQT
from QTdata import BO_data_write

job = 'w_config'
# Optimization Parameters
# rbf = ConstantKernel(1.0) * RBF(length_scale_bounds=(0.01, 0.5)) # .107
m52 = ConstantKernel(1.0) * Matern(nu=2.5, length_scale_bounds=(0.01, 0.5))
gpr = GaussianProcessRegressor(kernel=m52, n_restarts_optimizer=2)
num_init = 50
num_itr = 200
T = np.pi / (2 * 0.125)
dim = (7,3) # dimension of the problem
s, d = dim
w_bound = [(0.125,12.5)] * (s-2) # for w as the unit of V=0.125
x_bound = [(-0.9999, 0.9999)] * ((s-2)*d)
bound = x_bound + w_bound
inv_G_ls = [0.35938137*T]#[2.58*10**(-3)*T]
repeat = 1



for G, inv_G in enumerate(inv_G_ls):
    y = lambda site_w: OpenQT(s,d,np.array(site_w[:(s-2)*d]),np.array(site_w[(s-2)*d:])
                              ,Gamma=1/inv_G,n_p=3).T_r(epabs=0.001)[0] # object function
    note = "inv_Gamma = {0}T".format(inv_G/T)
    print(note)
    filename = '{0}s_{1}d_job_{2}_{3}invG_kappa_0.01.csv'.format(s,d,job,inv_G/T)
    for itr in range(repeat):
        lhs = Lhs(lhs_type="classic", criterion=None)
        X_init = lhs.generate(bound, num_init)
        print(X_init)
        Y_init = np.array([y(X_i) for X_i in X_init])
        print(Y_init)
             
        # Run BO
        r = gp_minimize(y,                   # negative for maximize; positive for minimize
                        bound,
                        base_estimator=gpr, 
                        acq_func='LCB',      # expected improvement
                        kappa = 0.01,
                        # xi=0.01,          # exploitation-exploration trade-off
                        # acq_optimizer="sampling", # for the periodic kernel 
                        n_calls=num_itr,         # number of iterations (s-2)*d*100
                        n_random_starts=-num_init,   # initial samples are provided
                        n_restarts_optimizer = 5,
                        x0=X_init,  # initial samples
                        y0=Y_init.ravel(),   # negative for maximize; positive for minimize
                        verbose=True)
        
        # Print results
        res = r.fun
        r.x_iters = np.array(r.x_iters)
        pos = np.atleast_2d(r.x).ravel()
        print("The optimal energy level is w={0}, and the position is {1}, with the minimum transfer time {2:.3f}T".format(pos[(s-2)*d:], pos[:(s-2)*d],res/T))
        print("Transfer time without Bath is {0}T".format(NCohQT(s,d,pos[:(s-2)*d],1/inv_G).T_r()[0]/T))
        BO_data_write(r,filename,[T,inv_G,itr,pos[:(s-2)*d]],'w')
        
    


