# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 11:42:27 2025

@author: kazahme
"""

import gurobipy as gp, numpy as np
from gurobipy import GRB
import pandas as pd

buses = pd.read_excel('powerSystem.xlsx',sheet_name = 'bus')
generations = pd.read_excel('powerSystem.xlsx',sheet_name = 'generation')
loads = pd.read_excel('powerSystem.xlsx',sheet_name = 'load')
lines = pd.read_excel('powerSystem.xlsx',sheet_name = 'line')
Nb = len(buses)
Ng = len(generations)
Nd = len(loads)
Nl = len(lines)
Bl = 1000

model = gp.Model('Power Flow Optimization Model')

# Here the objective function is a constent value*decision variable, so Linear Prorgramming Problem

# Add Variables
Pg = model.addVars(range(Ng),lb = 0.0, ub={g : generations.pgmax[g] for g in generations.index}, name = 'Pg')
Pl = model.addVars(range(Nl), lb={l : -lines.plmax[l] for l in lines.index},ub={l : lines.plmax[l] for l in lines.index}, name = 'Pl')
theta = model.addVars(range(Nb),lb = -np.pi, ub = np.pi)

# Objective Function
model.setObjective(gp.quicksum([Pg[i]*generations.cost[i] for i in range(Ng)]), GRB.MINIMIZE)

# Constraints
for n in buses.index:
    sum_Pg = gp.quicksum([Pg[g] for g in generations.index[generations.bus == n]])
    sum_Pls = gp.quicksum([Pl[l] for l in lines.index[lines.from_bus==n]])
    sum_Plr = gp.quicksum([Pl[l] for l in lines.index[lines.to_bus==n]])
    sum_Pd = gp.quicksum([loads.load[d] for d in loads.index[loads.bus == n]])
    model.addConstr(sum_Pg - sum_Pls + sum_Plr == sum_Pd, name = f"balance[{n}]")
    
for l in lines.index:
    n_send = lines.from_bus[l]
    n_rec = lines.to_bus[l]
    delta_theta = theta[n_send] - theta[n_rec]
    model.addConstr(Pl[l] == Bl*delta_theta, name=f"flow[{l}]")
    
# setting the reference bus (slack bus) angle in your DC power flow model.
model.addConstr(theta[0] == 0, name='slack')

# Solve the model
model.optimize()
model.write('optimal_power_flow_model.lp')
#model.write('case_study_model.mps')

# Now, read and print the contents of the file
with open("optimal_power_flow_model.lp", "r") as f:
    print(f.read())
    
# ==== Results ====
if model.status == GRB.OPTIMAL:
    print("Optimal objective value:", model.objVal)
    for v in model.getVars():
        print(f"{v.varName} = {v.x}")
        
import pandas as pd

data = [(v.varName, v.x) for v in model.getVars() if abs(v.x) > 1e-6]
df = pd.DataFrame(data, columns=["Variable", "Value"])
print(df)
    
    
