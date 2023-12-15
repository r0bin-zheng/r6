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
from evolution.multi_tdeadp_3 import scalar_dom_ea_dp as scalar_dom_ea_dp_3
from evolution.multi_tdeadp_4 import scalar_dom_ea_dp as scalar_dom_ea_dp_4
from evolution.algorithms import scalar_dom_ea_dp
from evolution.counter import PerCounter
from evolution.selection import pareto_scalar_nn_filter, nondominated_filter, pareto_nn_filter_2, pareto_nn_filter_3
from evolution.ranking import non_dominated_ranking
from evolution.dom import pareto_dominance
from evolution.visualizer import draw_igd_curve, draw_parato
from evolution.phase_strategy import PhaseStrategy, strategy_list, all_phase_list

from learning.model_init import init_dom_nn_classifier, init_kriging_model, init_rbf_model, init_kpls_model
from learning.model_update import update_dom_nn_classifier, update_kriging_model

from problems.factory import get_problem, get_problem_pymoo
from problems.rp import get_reference_points
from problems.metrics import get_igd, get_igd_pymoo, get_gd_pymoo, get_hv_pymoo, get_gdplus_pymoo, get_igdplus_pymoo, get_pareto_front, get_igd_pf, get_gd_pf, get_hv_pf, get_gdplus_pf, get_igd_plus_pf
from pymoo.util.ref_dirs import get_reference_directions


EXP_SAVE_PATH = "./exp/"


class Exp:
    """Experiment class"""
    def __init__(self, id, n_var, n_obj, alg, max_fe, problem, phase_list, strategy, rate):

        # exp info
        self.id = id
        self.time = time.localtime()
        self.path = EXP_SAVE_PATH + self.id

        # problem info
        self.problem_name = problem
        self.n_var = n_var
        self.n_obj = n_obj

        # algorithm info
        self.alg = alg
        self.max_fe = max_fe
        self.phase_list = phase_list
        self.strategy = strategy
        self.rate = rate  # the rate of first phase

        # visualization
        self.alpha = 0.1 if self.problem_name == "dtlz7" else 0.3

    def init(self):
        self.path_init()
        self.toolbox_init()

    def run(self):
        self.print_info()
        self.start_time = time.time()
        self.__run__()
        self.end_time = time.time()
        self.get_performance_indicator()
        self.save()

    def path_init(self):
        if not os.path.exists(self.path):
            os.mkdir(self.path)

    def toolbox_init(self):

        # problem
        # define a problem to be solved
        self.problem = get_problem(
            self.problem_name, n_var=self.n_var, n_obj=self.n_obj)
        self._problem = get_problem_pymoo(
            self.problem_name, n_var=self.problem.n_var, n_obj=self.problem.n_obj)  # problem for pymoo

        # ref_points
        # define a set of structured weight vectors
        self.ref_points = get_reference_points(self.problem.n_obj)

        # alg args
        self.mu = len(self.ref_points)  # population size
        self.init_size = 11 * self.problem.n_var - 1  # the number of initial solutions
        self.lambda_ = 7000    # the number of offsprings
        self.cxpb = 1.0       # crossover probability
        self.mutpb = 1.0      # mutation probability
        self.dic = 30                    # distribution index for crossover
        self.dim = 20                    # distribution index for mutation
        self.pm = 1.0 / self.problem.n_var    # mutation probability
        self.lr = 0.001      # learning rate
        self.wdc = 0.00001   # weight decay coefficient
        self.hidden_size = 200      # the number of units in each hidden layer
        self.num_layers = 2         # the number of hidden layers
        self.epochs = 20             # epochs for initiating FNN
        self.batch_size = 32         # min-batch size for training
        self.acc_thr = 0.9           # threshold for accuracy
        # the maximum number of solutions used in updating
        self.window_size = 11 * self.problem.n_var + 24
        # the maximum size in each category
        self.category_size = 300
        self.ps = PhaseStrategy(self.phase_list, change_strategy=self.strategy,
                                rate=self.rate, max_evaluations=(self.max_fe-self.init_size))

        # create types for the problem, refer to DEAP documentation
        creator.create("FitnessMin", base.Fitness,
                       weights=(-1.0,) * self.problem.n_obj)
        creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()

        # customize the population initialization
        self.toolbox.register("population", lhs_init_population,
                              list, creator.Individual, self.problem.xl, self.problem.xu)

        # customize the function evaluation
        self.toolbox.register("evaluate", self.problem.evaluate)

        # customize the crossover operator, SBX is used here
        self.toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=list(
            self.problem.xl), up=list(self.problem.xu), eta=30.0)

        # customize the mutation operator, polynomial mutation is used here
        self.toolbox.register("mutate", tools.mutPolynomialBounded, low=list(self.problem.xl), up=list(self.problem.xu), eta=20.0,
                              indpb=1.0 / self.problem.n_var)

        # customize the variation method for producing offsprings, genetic variation is used here
        self.toolbox.register("variation", random_genetic_variation,
                              toolbox=self.toolbox, cxpb=self.cxpb, mutpb=self.mutpb)

        # customize the survival selection
        self.toolbox.register("select", sel_scalar_dea)

        # customize the survival selection
        self.toolbox.register("select_local", sel_scalar_dea_parato)

        # the cluster operator in Theta-DEA
        self.toolbox.register("cluster_scalarization",
                              cluster_scalarization, ref_points=self.ref_points)

        # normalize the decision variables for training purpose
        self.toolbox.register(
            "normalize_variables", var_normalization, low=self.problem.xl, up=self.problem.xu)

        # if GPU is available, use GPU, else use CPU
        self.device = torch.device(
            'cuda:3' if torch.cuda.is_available() else 'cpu')

        # customize the initiation of Pareto-Net
        self.toolbox.register("init_pareto_model", init_dom_nn_classifier,
                              device=self.device, input_size=2 * self.problem.n_var, hidden_size=self.hidden_size,
                              num_hidden_layers=self.num_layers, batch_size=self.batch_size, epochs=self.epochs,
                              activation='relu', lr=self.lr, weight_decay=self.wdc)

        # customize the initiation of Theta-Net
        self.toolbox.register("init_scalar_model", init_dom_nn_classifier,
                              device=self.device, input_size=2 * self.problem.n_var, hidden_size=self.hidden_size,
                              num_hidden_layers=self.num_layers, batch_size=self.batch_size, epochs=self.epochs,
                              activation='relu', lr=self.lr, weight_decay=self.wdc)

        # customize the updating of Pareto-Net
        self.toolbox.register("update_pareto_model", update_dom_nn_classifier, device=self.device, max_window_size=self.window_size,
                              max_adjust_epochs=self.epochs, batch_size=self.batch_size, lr=self.lr, acc_thr=self.acc_thr, weight_decay=self.wdc)

        # customize the updating of Theta-Net
        self.toolbox.register("update_scalar_model", update_dom_nn_classifier, device=self.device, max_window_size=self.window_size,
                              max_adjust_epochs=self.epochs, batch_size=self.batch_size, lr=self.lr, acc_thr=self.acc_thr, weight_decay=self.wdc)

        # two-stage preselection,
        # if just want to obtain solutions, disable "visualization" since it will slow the program
        self.toolbox.register("filter", pareto_scalar_nn_filter, device=self.device, ref_points=self.ref_points,
                              counter=PerCounter(len(self.ref_points)), toolbox=self.toolbox, visualization=True, id=self.id)

        # self.toolbox.register("filter_parato", pareto_nn_filter_2, device=self.device)
        self.toolbox.register("filter_parato", pareto_nn_filter_3, device=self.device, ref_points=self.ref_points,
                              counter=PerCounter(len(self.ref_points)), toolbox=self.toolbox, visualization=True, id=self.id)

        self.toolbox.register("next", self.ps.next)

    def print_info(self):
        print("--------------------------------------Info--------------------------------------")
        print("ID: " + str(self.id))
        print("Path: " + str(self.path))
        print("Time: " + str(time.strftime("%Y-%m-%d %H:%M:%S", self.time)))
        print("--------------------------------------Problem--------------------------------------")
        print("Problem: " + str(self.problem_name))
        print("Number of objects: " + str(self.n_obj))
        print("Number of variables: " + str(self.n_var))
        print("--------------------------------------Algorithm--------------------------------------")
        print("Algorithm choice: " + str(self.alg))
        print("Phase list: " + str(self.phase_list))
        print("Phase strategy: " + str(strategy_list[self.strategy]))
        print("First phase rate: " + str(self.rate))
        print("Population size: " + str(self.mu))
        print("Maximum number of function evaluations: " + str(self.max_fe))
        print("--------------------------------------Output--------------------------------------")

    def get_performance_indicator(self):
        self.non_dom_solutions = non_dominated_ranking(
            self.archive, pareto_dominance)[0]
        self.non_dom_solutions_y = self.toolbox.evaluate(
            self.non_dom_solutions)
        self.front = get_pareto_front(self._problem)

        self.igd = get_igd_pf(self.non_dom_solutions, self.front)
        self.gd = get_gd_pf(self.non_dom_solutions, self.front)
        self.hv = get_hv_pf(self.non_dom_solutions, self.front)
        self.time_cost = self.end_time - self.start_time

        print("*" * 80)
        print("IGD:       " + str(self.igd))
        print("GD:        " + str(self.gd))
        print("HV:        " + str(self.hv))
        print("Time cost: ", self.time_cost)
        print("\nFinal Pareto nondominated solutions obtained:")
        for ind in self.non_dom_solutions:
            print(ind.fitness.values)

    def save(self):
        self.save_result()
        self.draw_igd_curve()
        self.draw_parato()

    def save_result(self):
        with open(self.path + '/result.txt', 'w') as f:

            str_time = time.strftime("%Y-%m-%d %H:%M:%S", self.time)
            phases = self.ps.get_phases()

            f.write(
                f"--------------------------------------Info--------------------------------------\n")
            f.write(f"ID: {self.id}\n")
            f.write(f"Path: {self.path}\n")
            f.write(f"Time: {str_time}\n")
            f.write(f"Success: {self.success}\n")

            f.write(
                f"--------------------------------------Problem--------------------------------------\n")
            f.write(f"Problem: {self.problem_name}\n")
            f.write(f"Number of objects: {self.n_obj}\n")
            f.write(f"Number of variables: {self.n_var}\n")

            f.write(
                f"--------------------------------------Algorithm--------------------------------------\n")
            f.write(f"Algorithm choice: {self.alg}\n")
            f.write(f"Phase list: {self.phase_list} ({phases})\n")
            f.write(f"Phase strategy: {strategy_list[self.strategy]}\n")
            f.write(f"First phase rate: {self.rate}\n")
            f.write(f"Population size: {self.mu}\n")
            f.write(f"Maximum number of function evaluations: {self.max_fe}\n")

            f.write(
                f"--------------------------------------Result--------------------------------------\n")
            f.write(f"IGD: {self.igd}\n")
            f.write(f"GD: {self.gd}\n")
            f.write(f"HV: {self.hv}\n")
            f.write(f"Time cost: {self.end_time - self.start_time}\n\n")

            f.write(f"non_dom_solutions:\n")
            for item in self.non_dom_solutions:
                f.write(str(item) + '\n')
            f.write(f"\nnon_dom_solutions_y:\n")
            for item in self.non_dom_solutions_y:
                f.write(str(item) + '\n')

    def draw_igd_curve(self):
        igd_arr = []
        for _archive in self.archive_arr:
            sln = non_dominated_ranking(_archive, pareto_dominance)[0]
            igd_arr.append(get_igd_pf(sln, self.front))
            if len(igd_arr) == 1:
                for i in range(self.init_size - 1):
                    igd_arr.append(igd_arr[0])

        with open(self.path + '/igd.txt', 'w') as f:
            for i in igd_arr:
                f.write(str(i) + '\n')

        draw_igd_curve(igd_arr, self.id)

    def draw_parato(self):
        archive_y = self.toolbox.evaluate(self.archive)
        if self.problem.n_obj == 2 or self.problem.n_obj == 3:
            draw_parato(np.array(archive_y), self.front, self.id,
                        alpha=self.alpha, name="archive")
            draw_parato(np.array(self.non_dom_solutions_y), self.front,
                        self.id, alpha=self.alpha, name="non_dom_solutions")

    def __run__(self):
        success = True
        if self.alg == 0:
            archive, archive_arr = scalar_dom_ea_dp(self.init_size, self.toolbox, self.mu, self.lambda_, self.max_fe,
                                                    category_size=self.category_size)
        elif self.alg == 1:
            archive, archive_arr = scalar_dom_ea_dp_1(self.init_size, self.toolbox, self.mu, self.lambda_, self.max_fe,
                                                      category_size=self.category_size, rate=self.rate)
        elif self.alg == 2:
            archive, archive_arr = scalar_dom_ea_dp_2(self.init_size, self.toolbox, self.mu, self.lambda_, self.max_fe,
                                                      category_size=self.category_size, rate=self.rate, strategy=self.strategy)
        elif self.alg == 3:
            archive, archive_arr = scalar_dom_ea_dp_3(self.init_size, self.toolbox, self.mu, self.lambda_, self.max_fe,
                                                      category_size=self.category_size, rate=self.rate, strategy=self.strategy)
        elif self.alg == 4:
            archive, archive_arr, success = scalar_dom_ea_dp_4(self.init_size, self.toolbox, self.mu, self.lambda_, self.max_fe,
                                                               category_size=self.category_size)
        self.archive = archive
        self.archive_arr = archive_arr
        self.success = success

    def __str__(self):
        return f"ID: {self.id}\n" \
               f"Number of objects: {self.n_obj}\n" \
               f"Number of variables: {self.n_var}\n" \
               f"Algorithm choice: {self.alg}\n" \
               f"Maximum number of function evaluations: {self.max_fe}\n" \
               f"Problem name: {self.problem_name}\n" \
               f"Phase list: {self.phase_list}\n" \
               f"Phase strategy: {strategy_list[self.strategy]}\n" \
               f"First phase rate: {self.rate}\n"


def get_id():
    return time.strftime("%Y%m%d%H%M%S", time.localtime())


parser = argparse.ArgumentParser(description="")
parser.add_argument('--n_obj', type=int, required=True,
                    help='Number of objects')
parser.add_argument('--n_var', type=int, required=True,
                    help='Number of variables')
parser.add_argument('--alg', type=int, required=True, help='Algorithm choice')
parser.add_argument('--phase_list', type=int, required=True,
                    help='Algorithm phase list')
parser.add_argument('--strategy', type=int, required=True,
                    help='Algorithm phase strategy')
parser.add_argument('--rate', type=float, required=True,
                    help='first phase rate')
parser.add_argument('--max_fe', type=int, required=True,
                    help='Maximum number of function evaluations')
parser.add_argument('--problem', type=str, required=True, help='Problem name')
parser.add_argument('--id', type=str, required=True,
                    help='Exp id', default="0")
args = parser.parse_args()

id_ = args.id if args.id != "0" else get_id()
exp = Exp(id=id_, n_var=args.n_var, n_obj=args.n_obj, alg=args.alg, max_fe=args.max_fe, problem=args.problem,
          phase_list=args.phase_list, strategy=args.strategy, rate=args.rate)
exp.init()
exp.run()
