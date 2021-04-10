# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 11:03:54 2020
varied of site, w, g_t

@author: HBK
"""

from pymoo.algorithms.so_cmaes import CMAES
from pymoo.optimize import minimize
from pymoo.model.problem import FunctionalProblem
from pymoo.util.termination.default import MultiObjectiveDefaultTermination
from QTransport_cls import NCohQT
from QTransport_cls import OpenQT
import numpy as np
import pandas as pd
from constraint_v2 import constr
# -----------------------------------------------------------------------------
# Parameter initialization:
job="constraint_analysis"
T = np.pi / (2 * 0.125)
#inv_G_ls = np.linspace(0.5,5,10) * T
inv_G_ls = [0.001*T]#np.logspace(-2, 0, 10)[:5] * T


for num, inv_G in enumerate(inv_G_ls):
    dim = (3,2) # dimension of the problem
    s, d = dim
    N = (s-2)*d
    repeat = 1
    r_min = 0.1

    # -----------------------------------------------------------------------------
    # Optimization parameters:
    termination = MultiObjectiveDefaultTermination(
        x_tol=1e-8,
        f_tol=10-8,
        n_max_gen=10) # max=300
    
    objs = [lambda site_w_gt: OpenQT(s,d,np.array(site_w_gt[:(s-2)*d]),np.array(site_w_gt[(s-2)*d:-1])
                              ,Gamma=1/inv_G,n_p=3,g_t=site_w_gt[-1]).T_r(epabs=0.01,limit=100)[0]]
    
    constr_ieq = constr(s,d,r_min,bc=True)
    
    
    # -----------------------------------------------------------------------------
    # Opimization loop and file saving
    for itr in range(repeat):
        problem = FunctionalProblem(N + s-2 +1,
                                    objs,
                                    constr_ieq=constr_ieq,
                                    xl=np.concatenate((-np.ones(N), 0.125*np.ones(s-2), 1*np.ones(1))),
                                    xu=np.concatenate((np.ones(N), 12.5*np.ones(s-2), 30*np.ones(1)))) # sigma = 0.001
        
        # increase population size for each itr
        algorithm = CMAES(#x0 = x0,
                          sigma=10 * 10**(-3),
                          restarts=2)
        
        r = minimize(problem,
                        algorithm,
                        termination,
                        save_history=True,
                        verbose=True)
        
        n_evals = []    # corresponding number of function evaluations
        y = []          # the objective space values in each generation
        cv = []         # constraint violation in each generation
        X = []
        filename = '{0}s_{1}d_job_{2}_{3}T_invG_kappa_0.01.csv'.format(s,d,job,inv_G/T)
        # iterate over the deepcopies of algorithms
        for algorithm in r.history:
        
            # store the number of function evaluations
            n_evals.append(algorithm.evaluator.n_eval)
        
            # retrieve the optimum from the algorithm
            opt = algorithm.opt
        
            # store the least contraint violation in this generation
            cv.append(opt.get("CV").min())
            
            # filter out only the feasible and append
            feas = np.where(opt.get("feasible"))[0]
            y.append(algorithm.opt.get('F').min(axis=0)[0]/T)
            X.append(algorithm.opt.get('X')[0].tolist())
            
        #np.savetxt("pos.csv", pos, delimiter=",")
        #np.savetxt("res.csv", res, delimiter=",")
        
        # saving files
        print("The optimal position is at r = {0}, with the minimum tansfer time t = {1:.3f}T".format(r.X, r.F[0]/T))
        res = {
                'Gamma': "1/Gamma = {0}T".format(inv_G/T),
                'Best_pos':[r.X],
                'Best_val(T)':r.F[0]/T,
                }
        data_res = pd.DataFrame(res)
        
        eva = {
                'pos':X,
                'res(T)':y, 
                }
        data_eva = pd.DataFrame(eva)
        data_eva.index.name = 'Itr'
        data_eva.index = data_eva.index + 1
        
        note = "1/Gamma = {0}T".format(inv_G/T)
        pd.Series([note, "itr = {0}".format(itr+1)]).to_csv('Temporary_results\\ES_T_evaluation_{0}.csv'.format(filename), mode='a')
        data_eva.to_csv('Temporary_results\\ES_T_evaluation_{0}.csv'.format(filename), mode='a')
        data_res.to_csv('Temporary_results\\ES_T_results_{0}.csv'.format(filename), mode='a')
