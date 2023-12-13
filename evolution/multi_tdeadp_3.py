from evolution.dom import scalar_dominance, pareto_dominance, epsilon_dominance
from evolution.utils import *
from evolution.ranking import non_dominated_ranking
from problems.metrics import get_igd, get_igd_pymoo

import numpy as np


# Main loop of Theta-DEA-DP

# 主循环
def scalar_dom_ea_dp(init_size, toolbox, mu, lambda_, max_evaluations,
                     category_size=300, rate=0.5, strategy=2, f_min=None, f_max=None, exp_path=None):
    
    # 初始化种群
    pop = toolbox.population(init_size)   # initialize the population
    toolbox.normalize_variables(pop)   # normalize all decision variables to [-1, 1]

    # 上下界
    f_min, f_max = init_obj_limits(f_min, f_max)   # read the limits of the objectives

    # 评估初始样本
    full_evaluate(pop, toolbox, f_min, f_max)     # evaluate all the solutions in the initial population
    evaluations = init_size

    # 归档
    archive = []
    archive_data = []
    archive.extend(pop)    # add the evaluated solutions to archive
    archive_data.append(archive.copy())

    rep_individuals = init_scalar_rep(pop)      # initialize the scalar (theta) representative solutions

    # get non-dominated ones among the scalar (theta) reps
    nd_rep_individuals = get_non_dominated_scalar_rep(rep_individuals)

    # 保存优势关系
    # p_rel_map and s_rel_map are used to save the dominance relation between evaluated solutions,
    # avoid repetitive computation
    p_rel_map, s_rel_map = init_dom_rel_map(max_evaluations)

    print("Initiating Pareto-Net:")
    p_model = toolbox.init_pareto_model(archive, p_rel_map, epsilon_dominance)  # init Pareto-Net

    print("Initiating Theta-Net:")
    s_model = toolbox.init_scalar_model(archive, s_rel_map, scalar_dominance)  # init Theta-Net

    # print("p_rel_map", p_rel_map)
    # print("s_rel_map", s_rel_map)

    global_flag = True if strategy == 2 else False
    one_time_global = False
    run_time = max_evaluations - evaluations
    r = rate
    change_flag = run_time * r + evaluations

    while evaluations < max_evaluations:
        if evaluations > change_flag:
            global_flag = False if strategy == 2 else True
        else:
            global_flag = True if strategy == 2 else False

        print("Eval: " + str(evaluations) + ", go " + ("global" if global_flag else ("one time global" if one_time_global else "local")) + " search.")

        offsprings = toolbox.variation(pop, lambda_) # produce offspring using genetic operations
        toolbox.normalize_variables(offsprings)

        if one_time_global or global_flag:
            # use two-stage preselection to select a solution for function evaluation
            individual = toolbox.filter(offsprings, rep_individuals, nd_rep_individuals, p_model, s_model, category_size, evalTimes=evaluations)      
        else:
            # individual = toolbox.filter_parato(offsprings, p_model)
            individual = toolbox.filter_parato(offsprings, rep_individuals, nd_rep_individuals, p_model, s_model, category_size, evalTimes=evaluations)   
            
            # print("Initiating Local Model:")
            # toolbox.update_local_model(local_model, archive)  # update local model
            # individual = toolbox.filter_local(offsprings, local_model)

        if individual is None:
            if global_flag == False:
                one_time_global = True
            continue

        one_time_global = False

        # 评估新个体的目标值，以及在聚类中的scalar_dist
        full_evaluate([individual], toolbox, f_min, f_max)  # evaluate the selected solution
        evaluations += 1

        archive.append(individual)

        # update representative solutions
        if update_scalar_rep(rep_individuals, individual):
            nd_rep_individuals = get_non_dominated_scalar_rep(rep_individuals)

        # # truncate the population size to mu
        # if global_flag:
        #     pop = toolbox.select(pop + [individual], mu)
        # else:
        #     pop = toolbox.select_local(pop + [individual], mu)

        # truncate the population size to mu
        pop = toolbox.select(pop + [individual], mu)

        # if global_flag:
        #     # update Theta-Net
        #     if s_model is None:
        #         print("Initiating Theta-Net:")
        #         s_model = toolbox.init_scalar_model(archive, s_rel_map, scalar_dominance)
        #     else:
        #         print("Theta-Net is updating:")
        #         toolbox.update_scalar_model(s_model, archive, s_rel_map, scalar_dominance)
        # else:
        #     # update Pareto-Net
        #     if p_model is None:
        #         print("Initiating Pareto-Net:")
        #         p_model = toolbox.init_pareto_model(archive, p_rel_map, pareto_dominance)
        #     else:
        #         print("Pareto-Net is updating:")
        #         toolbox.update_pareto_model(p_model, archive, p_rel_map, pareto_dominance)

        # update Pareto-Net
        if p_model is None:
            print("Initiating Pareto-Net:")
            p_model = toolbox.init_pareto_model(archive, p_rel_map, epsilon_dominance)
        else:
            print("Pareto-Net is updating:")
            toolbox.update_pareto_model(p_model, archive, p_rel_map, epsilon_dominance)

        # update Theta-Net
        if s_model is None:
            print("Initiating Theta-Net:")
            s_model = toolbox.init_scalar_model(archive, s_rel_map, scalar_dominance)
        else:
            print("Theta-Net is updating:")
            toolbox.update_scalar_model(s_model, archive, s_rel_map, scalar_dominance)

        # print the objective values of representative solutions
        print("Scalar (theta) representative solutions: ")
        for ind in rep_individuals.values():
            print(ind.fitness.values)
        print("Non-dominated ones among scalar (theta) representative solutions:")
        for ind in nd_rep_individuals.values():
            print(ind.fitness.values)

        # save archive_data
        archive_data.append(archive.copy())
        # print("archive_data", archive_data)
        
        print("*" * 80)

    return archive, archive_data
