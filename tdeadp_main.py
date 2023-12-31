import torch

import numpy as np

from deap import base, creator
from deap import tools

from problems.factory import get_problem
from problems.rp import get_reference_points
from problems.metrics import get_igd, get_igd_pymoo, get_gd_pymoo, get_hv_pymoo, get_gdplus_pymoo, get_igdplus_pymoo

from evolution.initialization import lhs_init_population
from evolution.variation import random_genetic_variation
from evolution.norm import var_normalization
from evolution.scalar import cluster_scalarization
from evolution.selection import sel_scalar_dea
from evolution.algorithms import scalar_dom_ea_dp
from evolution.counter import PerCounter
from evolution.selection import pareto_scalar_nn_filter
from evolution.ranking import non_dominated_ranking
from evolution.dom import pareto_dominance

from learning.model_init import init_dom_nn_classifier
from learning.model_update import update_dom_nn_classifier

# 根据日期时间生成id，包括年月日时分秒
import time
import datetime
def get_id():
    return time.strftime("%Y%m%d%H%M%S", time.localtime())

# 生成id
id = get_id()

# 在当前目录下创建文件夹，名称为id
import os
os.mkdir(id)

n_var = 2
n_obj = 3
problem = get_problem("dtlz7", n_var=n_var, n_obj=n_obj)  # define a problem to be solved
#problem = get_problem("zdt1", n_var=10)
#problem = get_problem("zdt2", n_var=10)

_problem = get_problem_pymoo(problem.name, n_var=problem.n_var, n_obj=problem.n_obj)

pf_path = "./pf/DTLZ1.2D.pf"     # the path to true Pareto front of the problem
#pf_path = "../pf/ZDT1.2D.pf"
#pf_path = "../pf/ZDT2.2D.pf"

ref_points = get_reference_points(problem.n_obj)  # define a set of structured weight vectors

MU = len(ref_points)  # population size
INIT_SIZE = 11 * problem.n_var - 1  # the number of initial solutions

MAX_EVALUATIONS = 250   # The maximum number of function evaluations, should be larger than INIT_SIZE


LAMBDA = 7000    # the number of offsprings
CXPB = 1.0       # crossover probability
MUTPB = 1.0      # mutation probability

DIC = 30                    # distribution index for crossover
DIM = 20                    # distribution index for mutation
PM = 1.0 / problem.n_var    # mutation probability

LR = 0.001      # learning rate
WDC = 0.00001   # weight decay coefficient

HIDDEN_SIZE = 200      # the number of units in each hidden layer
NUM_LAYERS = 2         # the number of hidden layers
EPOCHS = 20             # epochs for initiating FNN
BATCH_SIZE = 32         # min-batch size for training


ACC_THR = 0.9           # threshold for accuracy
WINDOW_SIZE = 11 * problem.n_var + 24      # the maximum number of solutions used in updating

CATEGORY_SIZE = 300                        # the maximum size in each category

# create types for the problem, refer to DEAP documentation
creator.create("FitnessMin", base.Fitness, weights=(-1.0,) * problem.n_obj)
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# customize the population initialization
toolbox.register("population", lhs_init_population, list, creator.Individual, problem.xl, problem.xu)

# customize the function evaluation
toolbox.register("evaluate", problem.evaluate)

# customize the crossover operator, SBX is used here
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=list(problem.xl), up=list(problem.xu), eta=30.0)

# customize the mutation operator, polynomial mutation is used here
toolbox.register("mutate", tools.mutPolynomialBounded, low=list(problem.xl), up=list(problem.xu), eta=20.0,
                 indpb=1.0 / problem.n_var)

# customize the variation method for producing offsprings, genetic variation is used here
toolbox.register("variation", random_genetic_variation, toolbox=toolbox, cxpb=CXPB, mutpb=MUTPB)

# customize the survival selection
toolbox.register("select", sel_scalar_dea)

# the cluster operator in Theta-DEA
toolbox.register("cluster_scalarization", cluster_scalarization, ref_points=ref_points)


# normalize the decision variables for training purpose
toolbox.register("normalize_variables", var_normalization, low=problem.xl, up=problem.xu)


# if GPU is available, use GPU, else use CPU
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


# 默认两层隐藏层，每层200个神经元，学习率0.001，权重衰减0.00001，批次大小32，迭代20次
# customize the initiation of Pareto-Net
toolbox.register("init_pareto_model", init_dom_nn_classifier,
                 device=device,
                 input_size=2 * problem.n_var, hidden_size=HIDDEN_SIZE,
                 num_hidden_layers=NUM_LAYERS,
                 batch_size=BATCH_SIZE, epochs=EPOCHS,
                 activation='relu',
                 lr=LR, weight_decay=WDC)

# customize the initiation of Theta-Net
toolbox.register("init_scalar_model", init_dom_nn_classifier,
                 device=device,
                 input_size=2 * problem.n_var, hidden_size=HIDDEN_SIZE,
                 num_hidden_layers=NUM_LAYERS,
                 batch_size=BATCH_SIZE, epochs=EPOCHS,
                 activation='relu',
                 lr=LR, weight_decay=WDC)

# customize the updating of Pareto-Net
toolbox.register("update_pareto_model", update_dom_nn_classifier, device=device, max_window_size=WINDOW_SIZE,
                 max_adjust_epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, acc_thr=ACC_THR, weight_decay=WDC)

# customize the updating of Theta-Net
toolbox.register("update_scalar_model", update_dom_nn_classifier, device=device, max_window_size=WINDOW_SIZE,
                 max_adjust_epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, acc_thr=ACC_THR, weight_decay=WDC)

# two-stage preselection,
# if just want to obtain solutions, disable "visualization" since it will slow the program
toolbox.register("filter", pareto_scalar_nn_filter, device=device, ref_points=ref_points,
                 counter=PerCounter(len(ref_points)), toolbox=toolbox, visualization=True, id=id)

start_time = time.time()

# run the algorithm and return all the evaluated solutions
archive = scalar_dom_ea_dp(INIT_SIZE, toolbox, MU, LAMBDA, MAX_EVALUATIONS, category_size=CATEGORY_SIZE)

end_time = time.time()

# obtain the Pareto non-dominated solutions
non_dom_solutions = non_dominated_ranking(archive, pareto_dominance)[0]

# # compute IGD
# # igd = get_igd(non_dom_solutions, pf_path)
# igd = get_igd_pymoo(non_dom_solutions, problem)


# print("Final Pareto nondominated solutions obtained:")
# Y = []
# for ind in non_dom_solutions:
#     Y.append(ind.fitness.values)
#     print(ind.fitness.values)

# # can choose to save the no-dominated solutions to a file
# # out_path = "xxx.txt"
# # np.savetxt(out_path, np.array(Y))
# print("*" * 80)
# print("IGD value for final solutions:")
# print(igd)

# print("Time cost: ", end_time - start_time)

# compute performance indicator
# igd = get_igd(non_dom_solutions, pf_path)
igd = get_igd_pymoo(non_dom_solutions, problem)
gd = get_gd_pymoo(non_dom_solutions, problem)
hv = get_hv_pymoo(non_dom_solutions, problem)

print("Final Pareto nondominated solutions obtained:")
Y = []
for ind in non_dom_solutions:
    Y.append(ind.fitness.values)
    print(ind.fitness.values)

# can choose to save the no-dominated solutions to a file
# out_path = "xxx.txt"
# np.savetxt(out_path, np.array(Y))
print("*" * 80)
print("IGD: " + str(igd))
print("GD:  " + str(gd))
print("HV:  " + str(hv))
print()

print("Time cost: ", end_time - start_time)

# draw the igd curve
igd_arr = []
for _archive in archive_arr:
    sln = non_dominated_ranking(_archive, pareto_dominance)[0]
    igd_arr.append(get_igd_pymoo(sln, problem))
    if len(igd_arr) == 1:
        for i in range(INIT_SIZE - 1):
            igd_arr.append(igd_arr[0])
draw_igd_curve(igd_arr, id)

# draw Scatter Plot
front = _problem.pareto_front()
if problem.n_obj == 2 or problem.n_obj == 3:
    draw_parato(np.array(archive), front, id)




