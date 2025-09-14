# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 00:13:42 2025

@author: kazahme
"""

import pyomo.environ as pyo, numpy as np, pandas as pd
from pyomo.environ import *
from pyomo.opt import SolverFactory
 
#input
buses = pd.read_excel('powerSystem.xlsx', sheet_name='bus')
generations = pd.read_excel('powerSystem.xlsx', sheet_name='generation')
loads = pd.read_excel('powerSystem.xlsx', sheet_name='load')
lines = pd.read_excel('powerSystem.xlsx', sheet_name='line')
Nb = len(buses)
Ng = len(generations)
Nd = len(loads)
Nl = len(lines)
 
#modelo
model = pyo.ConcreteModel()
model.Pg = pyo.Var(range(Ng))
model.Pl = pyo.Var(range(Nl))
model.theta = pyo.Var(range(Nb), bounds=(-np.pi,np.pi))
Pg = model.Pg
Pl = model.Pl
theta = model.theta
 
#objective function
model.obj = pyo.Objective(expr= sum([Pg[g]*generations.cost[g] for g in generations.index]), sense=minimize)
 
#power balance
model.balance = pyo.ConstraintList()
for n in buses.index:
    sum_Pg = sum([Pg[g] for g in generations.index[generations.bus==n]])
    sum_Pls = sum([Pl[l] for l in lines.index[lines.from_bus==n]])
    sum_Plr = sum([Pl[l] for l in lines.index[lines.to_bus==n]])
    sum_Pd = sum([loads.load[d] for d in loads.index[loads.bus==n]])
    model.balance.add(expr= sum_Pg - sum_Pls + sum_Plr == sum_Pd)
 
#power flow
model.flux = pyo.ConstraintList()
for l in lines.index:
    Bl = lines.Bl[l]
    n_send = lines.from_bus[l]
    n_rec = lines.to_bus[l]
    delta_theta = theta[n_send]-theta[n_rec]
    model.flux.add(expr= Pl[l] == Bl*delta_theta)
 
#generator limits
model.limger = pyo.ConstraintList()
for g in generations.index:
    model.limger.add(inequality(0,Pg[g],generations.pgmax[g]))
 
#power flow limits
model.limflux = pyo.ConstraintList()
for l in lines.index:
    model.limflux.add(inequality(-lines.plmax[l],Pl[l],lines.plmax[l]))
 
#reference
model.ref = pyo.Constraint(expr= theta[0] == 0)
 
#solve
opt = SolverFactory('glpk')
opt.solve(model)
 
#result
model.pprint()
model.obj()