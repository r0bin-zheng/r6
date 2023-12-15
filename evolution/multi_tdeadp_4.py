from evolution.dom import scalar_dominance, pareto_dominance, epsilon_dominance
from evolution.utils import *
from evolution.ranking import non_dominated_ranking
from problems.metrics import get_igd, get_igd_pymoo
from evolution.common import Timer

import numpy as np


# Main loop of Theta-DEA-DP
# 主循环
def scalar_dom_ea_dp(init_size, toolbox, mu, lambda_, max_evaluations,
                     category_size=300, f_min=None, f_max=None):
    
    # 初始化计时器
    timer = Timer()
    timer_filter = Timer()
    timer_net = Timer()
    timer.start(desc="评估初始样本")
    
    # 初始化种群
    pop = toolbox.population(init_size)   # initialize the population
    toolbox.normalize_variables(pop)   # normalize all decision variables to [-1, 1]

    # 上下界
    f_min, f_max = init_obj_limits(f_min, f_max)   # read the limits of the objectives

    # 评估初始样本
    full_evaluate(pop, toolbox, f_min, f_max)     # evaluate all the solutions in the initial population
    evaluations = init_size

    timer.next(desc="初始化归档")

    # 归档
    archive = []
    archive_data = []
    archive.extend(pop)    # add the evaluated solutions to archive
    archive_data.append(archive.copy())

    timer.next(desc="初始化代表解")

    rep_individuals = init_scalar_rep(pop)      # initialize the scalar (theta) representative solutions

    timer.next(desc="初始化非支配代表解")

    # get non-dominated ones among the scalar (theta) reps
    nd_rep_individuals = get_non_dominated_scalar_rep(rep_individuals)

    timer.next(desc="初始化优势关系")

    # 保存优势关系
    # p_rel_map and s_rel_map are used to save the dominance relation between evaluated solutions,
    # avoid repetitive computation
    p_rel_map, s_rel_map = init_dom_rel_map(max_evaluations)

    timer.next(desc="初始化Pareto-Net")

    print("Initiating Pareto-Net:")
    p_model = toolbox.init_pareto_model(archive, p_rel_map, epsilon_dominance, timer=timer_net)  # init Pareto-Net

    timer.next(desc="初始化Theta-Net")

    print("Initiating Theta-Net:")
    s_model = toolbox.init_scalar_model(archive, s_rel_map, scalar_dominance, timer=timer_net)  # init Theta-Net

    # print("p_rel_map", p_rel_map)
    # print("s_rel_map", s_rel_map)

    global_flag = toolbox.next() == "global"
    run_time = max_evaluations - evaluations
    prev_eval = evaluations
    repeat = 0
    success = True

    while evaluations < max_evaluations:
        if evaluations == prev_eval:
            repeat += 1
            if repeat == 50:
                success = False
                break
        else:
            repeat = 0
            prev_eval = evaluations
        
        timer.next(desc="生成子代")

        print("Eval: " + str(evaluations) + ", go " + ("global" if global_flag else "local") + " search.")

        offsprings = toolbox.variation(pop, lambda_) # produce offspring using genetic operations
        toolbox.normalize_variables(offsprings)

        timer.next(desc="过滤子代")

        if global_flag:
            # use two-stage preselection to select a solution for function evaluation
            individual = toolbox.filter(offsprings, rep_individuals, nd_rep_individuals,
                                        p_model, s_model, category_size, evalTimes=evaluations, timer=timer_filter)      
        else:
            individual = toolbox.filter_parato(offsprings, rep_individuals, nd_rep_individuals,
                                               p_model, s_model, category_size, evalTimes=evaluations, timer=timer_filter)   



        if individual is None:
            print("No solution is selected, go to next iteration.")
            global_flag = toolbox.next(False) == "global"
            continue
        else:
            global_flag = toolbox.next(True) == "global"

        timer.next(desc="评估候选个体的真实值")

        # 评估新个体的目标值，以及在聚类中的scalar_dist
        full_evaluate([individual], toolbox, f_min, f_max)  # evaluate the selected solution
        evaluations += 1

        archive.append(individual)

        timer.next(desc="更新代表解")

        # update representative solutions
        if update_scalar_rep(rep_individuals, individual):
            nd_rep_individuals = get_non_dominated_scalar_rep(rep_individuals)
        
        timer.next(desc="更新优势关系")

        # truncate the population size to mu
        pop = toolbox.select(pop + [individual], mu)

        timer.next(desc="更新Pareto-Net")

        # update Pareto-Net
        if p_model is None:
            print("Initiating Pareto-Net:")
            p_model = toolbox.init_pareto_model(archive, p_rel_map, epsilon_dominance, timer=timer_net)
        else:
            print("Pareto-Net is updating:")
            toolbox.update_pareto_model(p_model, archive, p_rel_map, epsilon_dominance, timer=timer_net)

        timer.next(desc="更新Theta-Net")

        # update Theta-Net
        if s_model is None:
            print("Initiating Theta-Net:")
            s_model = toolbox.init_scalar_model(archive, s_rel_map, scalar_dominance, timer=timer_net)
        else:
            print("Theta-Net is updating:")
            toolbox.update_scalar_model(s_model, archive, s_rel_map, scalar_dominance, timer=timer_net)

        timer.next(desc="计算IGD")

        # print the objective values of representative solutions
        print("Scalar (theta) representative solutions: ")
        for ind in rep_individuals.values():
            print(ind.fitness.values)
        print("Non-dominated ones among scalar (theta) representative solutions:")
        for ind in nd_rep_individuals.values():
            print(ind.fitness.values)

        # save archive_data
        archive_data.append(archive.copy())
        
        print("*" * 80)

    timer.end()
    timer.print_map()
    timer_filter.print_map()
    timer_net.print_map()
    
    return archive, archive_data, success
