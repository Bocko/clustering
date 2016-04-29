from pulp import *
from promethee import *
import csv
import pandas as pd
from scipy.cluster.vq import kmeans, vq
from statsmodels.sandbox.tools.tools_pca import pcasvd, pca
import matplotlib.pyplot as plt
from pca_kmeans_biplot import *

# criteria = ['Infrastructure', 'Spatial', 'Education', 'CultureEnv', 'Healthcare', 'Stability']
#
# weights = [0.15, 0.25, 0.075, 0.1875, 0.15, 0.1875]
#
# fctPrefCrit = [PreferenceType2(0), PreferenceType2(0), PreferenceType2(0),  PreferenceType2(0), PreferenceType2(0), PreferenceType2(0)]
#
# names = ['Hong Kong', 'Stockholm', 'Rome', 'New York', 'Atlanta', 'Buenos Aires', 'Santiago', 'Sao Paulo', 'Mexico City', 'New Delhi', 'Istanbul', 'Jakarta', 'Tehran', 'Dakar']
#
# eval_table = [[96.4, 75.0, 100.0, 85.9, 87.5, 95.0],
# [96.4, 58.9, 100.0, 91.2, 95.8, 95.0],
# [92.9, 67.3, 100.0, 91.7, 87.5, 80.0],
# [89.3, 65.2, 100.0, 91.7, 91.7, 70.0],
# [92.9, 42.9, 100.0, 91.7, 91.7, 85.0],
# [85.7, 42.3, 100.0, 85.9, 87.5, 70.0],
# [85.7, 35.1, 83.3, 89.1, 70.8, 75.0],
# [66.1, 52.4, 66.7, 80.3, 70.8, 60.0],
# [46.4, 65.8, 75.0, 82.4, 66.7, 45.0],
# [58.9, 58.6, 75.0, 55.6, 58.3, 55.0],
# [67.9, 47.5, 58.3, 68.8, 50.0, 55.0],
# [57.1, 42.3, 66.7, 59.3, 45.8, 50.0],
# [33.9, 53.6, 50.0, 35.9, 62.5, 50.0],
# [37.5, 22.6, 50.0, 59.7, 41.7, 50.0]]

# criteria = ['1', '2', '3', '4']
# weights = [0.25, 0.25, 0.25, 0.25]
# fctPrefCrit = [PreferenceType2(0), PreferenceType2(0), PreferenceType2(0),  PreferenceType2(0)]#
# eval_table = []
# with open('flower.csv', newline='') as csvfile:
#     content = csv.reader(csvfile, delimiter=',', quotechar='|')
#     for row in content:
#         eval_table.append(list(map(lambda x: float(x), row)))

criteria = ['crit1', 'crit2', 'crit3', 'crit4', 'crit5', 'crit6']
weights = [0.1, 0.2, 0.2, 0.2, 0.2, 0.1]
fctPrefCrit = [PreferenceType2(0), PreferenceType2(0), PreferenceType2(0),  PreferenceType2(0), PreferenceType2(0), PreferenceType2(0)]
eval_table = []
with open('shanghai.csv', newline='') as csvfile:
    content = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in content:
        eval_table.append(list(map(lambda x: float(x), row)))
eval_table = eval_table[:20]

# criteria = ['1', '2']
# weights = [0.5, 0.5]
# fctPrefCrit = [PreferenceType2(0), PreferenceType2(0)]
# eval_table = [[0, 0],
# [0.1, 0.1],
# [0.2, 0.2],
# [4, 4],
# [4.1, 4.1],
# [4.2, 4.2],
# [10, 10],
# [10.1, 10.1],
# [10.2, 10.2]]

uninetflows = uninetflows_eval(eval_table,criteria,weights,fctPrefCrit)

# uniposflows, uninegflows = uniflows_eval(candidats, criteres, init_weights, fctPrefCrit)

def netflow_eval(i,n,weights,gamma):
    return posflow_eval(i,n,weights,gamma) - negflow_eval(i,n,weights,gamma)

def posflow_eval(i,n,weights,gamma):
    crits = len(weights)
    posflow = 0
    for j in range(n):
        if j != i:
            for k in range(crits):
                posflow += weights[k] * gamma[(i,j,k)]
    return (1/(n-1)) * posflow

def negflow_eval(i,n,weights,gamma):
    crits = len(weights)
    negflow = 0
    for j in range(n):
        if j != i:
            for k in range(crits):
                negflow += weights[k] * gamma[(j,i,k)]
    return (1/(n-1)) * negflow

# Problem definition
prob = LpProblem("IPO_t",LpMaximize)

# Constants
M = 1e6
N = 4
EPS = 1e-6
#init_weights = [0.1, 0.3, 0.6]
#criteria = [str(x) for x in range(len(init_weights))]
n = len(eval_table)
m = len(eval_table[0])
alts = range(n)
crits = range(m)
clusts = range(N)

# Variables
c_comb = []
for i in alts:
    for h in clusts:
        c_comb.append((i,h))
c = LpVariable.dicts("c",c_comb,cat="Binary")

gamma_comb = []
for i in alts:
    for j in alts:
        if i != j:
            for k in crits:
                gamma_comb.append((i,j,k))
gamma = LpVariable.dicts("gamma",gamma_comb,cat="Binary")

alpha_comb = []
beta_comb = []
for i in alts:
    for j in alts:
        if i != j:
            for h in clusts:
                alpha_comb.append((i,j,h))
                for l in clusts:
                    if l != h:
                        beta_comb.append((i,j,h,l))
alpha = LpVariable.dicts("alpha",alpha_comb,cat="Binary")
beta = LpVariable.dicts("beta",beta_comb,cat="Binary")

mu_comb = []
nu_comb = []
for i in alts:
    for j in alts:
        if i != j:
            for h in clusts:
                for k in crits:
                    mu_comb.append((i,j,h,k))
                    for l in clusts:
                        if l != h:
                            nu_comb.append((i,j,h,l,k))
mu = LpVariable.dicts("mu",mu_comb,cat="Binary")
nu = LpVariable.dicts("nu",nu_comb,cat="Binary")

# Objective function
obj_hom = []
obj_het = []
for i in alts:
    for j in alts:
        if i != j:
            for k in crits:
                for h in clusts:
                    obj_hom.append(mu[(i,j,h,k)]*weights[k])
                    for l in clusts:
                        if l != h:
                            obj_het.append(nu[i,j,h,l,k]*weights[k])

# prob += lpSum(obj_hom)+lpSum(obj_het)
prob += lpSum(obj_het)-lpSum(obj_hom)

# Constraints
for i in alts:
    for j in alts:
        if i != j:
            for k in crits:
                prob += gamma[(i,j,k)] >= (eval_table[i][k]-eval_table[j][k])/M
                prob += gamma[(i,j,k)] <= (eval_table[i][k]-eval_table[j][k])/M + 1 - EPS
                for h in clusts[:-1]:
                    prob += mu[(i,j,h,k)] >= alpha[(i,j,h)] + gamma[(i,j,k)] -1
                    prob += mu[(i,j,h,k)] <= 0.5*(alpha[(i,j,h)] + gamma[(i,j,k)])

                    prob += alpha[(i,j,h)] >= c[(i,h)] + c[(j,h)] - 1
                    prob += alpha[(i,j,h)] <= 0.5*(c[i,h] + c[j,h])

                    prob += nu[(i,j,h,h+1,k)] >= beta[(i,j,h,h+1)] + gamma[(i,j,k)] - 1
                    prob += nu[(i,j,h,h+1,k)] <= 0.5*(beta[(i,j,h,h+1)] + gamma[(i,j,k)])

                    prob += beta[(i,j,h,h+1)] >= c[(i,h)] + c[(j,h+1)] - 1
                    prob += beta[(i,j,h,h+1)] <= 0.5*(c[(i,h)] + c[j,h+1])

                    # for l in clusts:
                    #     if l != h:
                    #         prob += nu[(i,j,h,l,k)] >= beta[(i,j,h,l)] + gamma[(i,j,k)] - 1
                    #         prob += nu[(i,j,h,l,k)] <= 0.5*(beta[(i,j,h,l)] + gamma[(i,j,k)])
                    #
                    #         prob += beta[(i,j,h,l)] >= c[(i,h)] + c[(j,l)] - 1
                    #         prob += beta[(i,j,h,l)] <= 0.5*(c[(i,h)] + c[j,l])

for h in clusts:
    prob += lpSum([c[(i,h)] for i in alts]) >= 1

for i in alts:
    prob += lpSum([c[i,h] for h in clusts]) == 1
#print(prob)

prob.writeLP("linprogclust.lp")
prob.solve(GUROBI())

for v in prob.variables():
    if 'c' in v.name:
        print(v.name, "=", v.varValue)
print("Objective function:", value(prob.objective))
print("Status:", LpStatus[prob.status])

clust_repart = []
f = open('clustering.m','w')
f.write("clusts = [ ")
for i in alts:
    for h in clusts:
        if c[(i,h)].varValue == 1:
            clust_repart.append(h)
            f.write(str(h) + '\n')
f.write("];\n")
f.write("criteria = {")
for crit in criteria:
    f.write("'" + crit + "' ")
f.write('};')
f.close()

uninetflows.insert(0,criteria)
with open("uninetflows.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(uninetflows)

df = pd.io.parsers.read_csv('uninetflows.csv')
data = df[criteria]
# data = (data - data.mean()) / data.std()
pca = pcasvd(data, keepdim=0, demean=False)
colors = ['gbyrk'[i] for i in clust_repart]
plt.figure(1)
biplot(plt, pca, labels=data.index, colors=colors,
       xpc=1, ypc=2)
plt.show()

# iter = 0
# sols = []
# while iter < 5:
#     prob.solve(pulp.GLPK())
#     print(LpStatus[prob.status])
#     sols.append(prob.variables())
#
#     iter += 1
