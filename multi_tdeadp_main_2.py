import argparse
import numpy as np
import os
import time
import torch

from deap import base, creator
from deap import tools

from enum import Enum

from evolution.initialization import lhs_init_population
from evolution.variation import random_genetic_variation
from evolution.norm import var_normalization
from evolution.scalar import cluster_scalarization
from evolution.selection import sel_scalar_dea, sel_scalar_dea_parato
from evolution.multi_tdeadp_1 import scalar_dom_ea_dp as scalar_dom_ea_dp_1
from evolution.multi_tdeadp_2 import scalar_dom_ea_dp as scalar_dom_ea_dp_2
from evolution.algorithms import scalar_dom_ea_dp
from evolution.counter import PerCounter
from evolution.selection import pareto_scalar_nn_filter, nondominated_filter, pareto_nn_filter_2, pareto_nn_filter_3
from evolution.ranking import non_dominated_ranking
from evolution.dom import pareto_dominance
from evolution.visualizer import draw_igd_curve, draw_parato

from learning.model_init import init_dom_nn_classifier, init_kriging_model, init_rbf_model, init_kpls_model
from learning.model_update import update_dom_nn_classifier, update_kriging_model

from problems.factory import get_problem, get_problem_pymoo
from problems.rp import get_reference_points
from problems.metrics import get_igd, get_igd_pymoo, get_gd_pymoo, get_hv_pymoo, get_gdplus_pymoo, get_igdplus_pymoo, get_pareto_front, get_igd_pf, get_gd_pf, get_hv_pf, get_gdplus_pf, get_igd_plus_pf 
from pymoo.util.ref_dirs import get_reference_directions


# 初始化文件

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument('--n_obj', type=int, required=True, help='Number of objects')
parser.add_argument('--n_var', type=int, required=True, help='Number of variables')
parser.add_argument('--alg', type=int, required=True, help='Algorithm choice')
parser.add_argument('--strategy', type=int, required=True, help='Algorithm phase strategy')
parser.add_argument('--rate', type=float, required=True, help='first phase rate')
parser.add_argument('--max_fe', type=int, required=True, help='Maximum number of function evaluations')
parser.add_argument('--problem', type=str, required=True, help='Problem name')
parser.add_argument('--id', type=str, required=True, help='Exp id', default="0")
args = parser.parse_args()

def get_id():
    return time.strftime("%Y%m%d%H%M%S", time.localtime())

if args.id == "0":
    id = get_id()
else:
    id = args.id
if not os.path.exists("exp/" + id):
    os.mkdir("exp/" + id)

# *****************************************************************************************

# 初始化参数

# ALG = 0 tdeap
# ALG = 1 tdeap_main_1
# ALG = 2 tdeap_main_2
ALG = args.alg

n_var = args.n_var
n_obj = args.n_obj
# dtlz1
problem = get_problem(args.problem, n_var=n_var, n_obj=n_obj)  # define a problem to be solved
_problem = get_problem_pymoo(problem.name, n_var=problem.n_var, n_obj=problem.n_obj)

print(problem.name)
print(_problem.name)

# print("problem dim", problem.n_var)
# print("problem xl", problem.xl)
# print("problem xu", problem.xu)

pf_path = "./pf/DTLZ1.2D.pf"     # the path to true Pareto front of the problem
# pf_path = "../pf/ZDT1.2D.pf"
# pf_path = "../pf/ZDT2.2D.pf"

ref_points = get_reference_points(problem.n_obj)  # define a set of structured weight vectors

MU = len(ref_points)  # population size
INIT_SIZE = 11 * problem.n_var - 1  # the number of initial solutions

MAX_EVALUATIONS = args.max_fe   # The maximum number of function evaluations, should be larger than INIT_SIZE

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
RATE = args.rate                           # the rate of first phase

# 0：global
# 1: local
# 2: global + local
# 3: local + global
STRATEGY = args.strategy

print("*" * 80)
print(f"ID: {args.id}")
print(f"Number of objects: {args.n_obj}")
print(f"Number of variables: {args.n_var}")
print(f"Algorithm choice: {args.alg}")
print(f"Maximum number of function evaluations: {args.max_fe}")
print(f"Problem name: {args.problem}")
print("*" * 80)

# *****************************************************************************************

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

# customize the survival selection
toolbox.register("select_local", sel_scalar_dea_parato)

# the cluster operator in Theta-DEA
toolbox.register("cluster_scalarization", cluster_scalarization, ref_points=ref_points)


# normalize the decision variables for training purpose
toolbox.register("normalize_variables", var_normalization, low=problem.xl, up=problem.xu)


# if GPU is available, use GPU, else use CPU
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


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

# toolbox.register("filter_parato", pareto_nn_filter_2, device=device)
toolbox.register("filter_parato", pareto_nn_filter_3, device=device, ref_points=ref_points,
                 counter=PerCounter(len(ref_points)), toolbox=toolbox, visualization=True, id=id)

# toolbox.register("filter_local", nondominated_filter)

# *****************************************************************************************

# 执行算法

start_time = time.time()

# run the algorithm and return all the evaluated solutions
if ALG == 0:
    archive, archive_arr = scalar_dom_ea_dp(INIT_SIZE, toolbox, MU, LAMBDA, MAX_EVALUATIONS, category_size=CATEGORY_SIZE)
elif ALG == 1:
    archive, archive_arr = scalar_dom_ea_dp_1(INIT_SIZE, toolbox, MU, LAMBDA, MAX_EVALUATIONS, category_size=CATEGORY_SIZE, rate=RATE)
elif ALG == 2:
    archive, archive_arr = scalar_dom_ea_dp_2(INIT_SIZE, toolbox, MU, LAMBDA, MAX_EVALUATIONS, category_size=CATEGORY_SIZE, rate=RATE, strategy=STRATEGY)

end_time = time.time()

# *****************************************************************************************

# 可视化、持久化结果

non_dom_solutions = non_dominated_ranking(archive, pareto_dominance)[0]
non_dom_solutions_y = toolbox.evaluate(non_dom_solutions)
front = get_pareto_front(_problem)

# compute performance indicator
# igd = get_igd(non_dom_solutions, pf_path)
igd = get_igd_pf(non_dom_solutions, front)
gd  = get_gd_pf(non_dom_solutions, front)
hv  = get_hv_pf(non_dom_solutions, front)

print("*" * 80)
print("IGD: " + str(igd))
print("GD:  " + str(gd))
print("HV:  " + str(hv))
print("Time cost: ", end_time - start_time)
print()

# 打印最终的非支配解
print("Final Pareto nondominated solutions obtained:")
Y = []
for ind in non_dom_solutions:
    Y.append(ind.fitness.values)
    print(ind.fitness.values)

# 保存结果到f'./exp/{id}/result.txt'
with open(f'./exp/{id}/result.txt', 'w') as f:
    strategy_str = {
        0: "global",
        1: "local",
        2: "global + local",
        3: "local + global"
    }

    f.write(f"ID: {id}\n\n")
    f.write(f"Problem: {problem.name}\n")
    f.write(f"Number of objects: {args.n_obj}\n")
    f.write(f"Number of variables: {args.n_var}\n\n")
    f.write(f"Algorithm choice: {args.alg}\n")
    f.write(f"Algorithm phase strategy: {strategy_str[args.strategy]}\n")
    f.write(f"First phase rate: {RATE}\n")
    f.write(f"Population size: {MU}\n")
    f.write(f"Maximum number of function evaluations: {args.max_fe}\n\n")
    f.write(f"IGD: {igd}\n")
    f.write(f"GD: {gd}\n")
    f.write(f"HV: {hv}\n")
    f.write(f"Time cost: {end_time - start_time}\n\n")
    f.write(f"non_dom_solutions:\n")
    for i in non_dom_solutions:
        f.write(str(i) + '\n')
    f.write(f"\nnon_dom_solutions_y:\n")
    for i in non_dom_solutions_y:
        f.write(str(i) + '\n')

# 结果可视化

# draw the igd curve
igd_arr = []
for _archive in archive_arr:
    sln = non_dominated_ranking(_archive, pareto_dominance)[0]
    igd_arr.append(get_igd_pf(sln, front))
    if len(igd_arr) == 1:
        for i in range(INIT_SIZE - 1):
            igd_arr.append(igd_arr[0])

# 保存igd_arr到f'./{id}/igd.txt'
with open(f'./exp/{id}/igd.txt', 'w') as f:
    for i in igd_arr:
        f.write(str(i) + '\n')
draw_igd_curve(igd_arr, id)

# draw Scatter Plot
# if problem.n_obj == 2:
#     front = _problem.pareto_front()
#     draw_parato(np.array(non_dom_solutions_y), front, id)
# elif problem.n_obj == 3:
#     success_front = True
#     try:
#         ref_dirs = get_reference_directions("uniform", 3, n_partitions=12)
#         front = _problem.pareto_front(ref_dirs=ref_dirs)
#     except:
#         success_front = False
#     if success_front == False:
#         front = _problem.pareto_front()
draw_parato(np.array(non_dom_solutions_y), front, id)
