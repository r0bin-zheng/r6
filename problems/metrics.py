import numpy as np

from pymoo.problems import get_problem
from pymoo.indicators.igd import IGD
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.gd import GD
from pymoo.indicators.gd_plus import GDPlus
from pymoo.util.ref_dirs import get_reference_directions

# compute igd value of the solutions obtained by the algorithm, pf_path refers the path to Pareto front


def get_igd(pop, pf_path):
    if len(np.shape(pop)) == 1:
        pop = [pop]

    front = np.loadtxt(pf_path)
    obj_values = np.array([ind.fitness.values for ind in pop])

    return compute_igd(obj_values, front)


def get_igd_pf(pop, front):
    if len(np.shape(pop)) == 1:
        pop = [pop]

    obj_values = np.array([ind.fitness.values for ind in pop])
    return compute_igd(obj_values, front)


def get_igd_plus_pf(pop, front):
    if len(np.shape(pop)) == 1:
        pop = [pop]

    obj_values = np.array([ind.fitness.values for ind in pop])
    return compute_igdplus(obj_values, front)


def get_gd_pf(pop, front):
    if len(np.shape(pop)) == 1:
        pop = [pop]

    obj_values = np.array([ind.fitness.values for ind in pop])
    return compute_gd(obj_values, front)


def get_gdplus_pf(pop, front):
    if len(np.shape(pop)) == 1:
        pop = [pop]

    obj_values = np.array([ind.fitness.values for ind in pop])
    return compute_gdplus(obj_values, front)


def get_hv_pf(pop, front):
    if len(np.shape(pop)) == 1:
        pop = [pop]

    obj_values = np.array([ind.fitness.values for ind in pop])
    return compute_hv(obj_values, front)


def get_igd_pymoo(pop, problem):
    if len(np.shape(pop)) == 1:
        pop = [pop]

    problem_ = get_problem(problem.name, n_var=problem.n_var, n_obj=problem.n_obj)
    front = get_pareto_front(problem_)
    obj_values = np.array([ind.fitness.values for ind in pop])

    return compute_igd(obj_values, front)


def compute_igd(obj_values, front):
    min_values = np.min(front, axis=0)
    max_values = np.max(front, axis=0)

    front = (front - min_values) / (max_values - min_values)
    obj_values = (obj_values - min_values) / (max_values - min_values)

    front = front[:, np.newaxis, :]
    front = np.repeat(front, repeats=len(obj_values), axis=1)

    obj_values = obj_values[np.newaxis, :, :]
    obj_values = np.repeat(obj_values, repeats=len(front), axis=0)

    dist_matrix = np.sqrt(np.sum((front - obj_values) * (front - obj_values), axis=2))

    min_dists = np.min(dist_matrix, axis=1)

    return np.average(min_dists)


def get_igdplus_pymoo(pop, problem):
    if len(np.shape(pop)) == 1:
        pop = [pop]

    problem_ = get_problem(problem.name, n_var=problem.n_var, n_obj=problem.n_obj)
    front = problem_.pareto_front()
    obj_values = np.array([ind.fitness.values for ind in pop])

    return compute_igdplus(obj_values, front)


def compute_igdplus(obj_values, front):
    ind = IGDPlus(front)
    return ind(obj_values)


def get_gd_pymoo(pop, problem):
    if len(np.shape(pop)) == 1:
        pop = [pop]

    problem_ = get_problem(problem.name, n_var=problem.n_var, n_obj=problem.n_obj)
    obj_values = np.array([ind.fitness.values for ind in pop])
    try:
        front = problem_.pareto_front()
    except Exception:
        pf_file_name = str(problem.name).upper() + "." + str(problem_.n_var) + "D.PF"
        front = np.loadtxt("./pf/" + pf_file_name)

    return compute_gd(obj_values, front)


def compute_gd(obj_values, front):
    ind = GD(front)
    return ind(obj_values)


def get_gdplus_pymoo(pop, problem):
    if len(np.shape(pop)) == 1:
        pop = [pop]

    problem_ = get_problem(problem.name, n_var=problem.n_var, n_obj=problem.n_obj)
    obj_values = np.array([ind.fitness.values for ind in pop])
    try:
        front = problem_.pareto_front()
    except Exception:
        pf_file_name = str(problem.name).upper() + "." + str(problem_.n_var) + "D.PF"
        front = np.loadtxt("./pf/" + pf_file_name)

    return compute_gdplus(obj_values, front)


def compute_gdplus(obj_values, front):
    ind = GDPlus(front)
    return ind(obj_values)


def get_hv_pymoo(pop, problem):
    if len(np.shape(pop)) == 1:
        pop = [pop]

    problem_ = get_problem(problem.name, n_var=problem.n_var, n_obj=problem.n_obj)
    obj_values = np.array([ind.fitness.values for ind in pop])
    try:
        front = problem_.pareto_front()
    except Exception:
        pf_file_name = str(problem.name).upper() + "." + str(problem_.n_var) + "D.PF"
        front = np.loadtxt("./pf/" + pf_file_name)

    return compute_hv(obj_values, front)


def compute_hv(obj_values, front):
    ind = Hypervolume(ref_point=np.max(front, axis=0) + 0.1)
    return ind(obj_values)


def get_pareto_front(problem):
    print(problem.name)
    print(str(problem.name))
    problem_ = get_problem(str(problem.name), n_var=problem.n_var, n_obj=problem.n_obj)
    try:
        if problem.n_obj > 3:
            # 如果目标维度大于3，获取ref_dirs
            n_partitions = problem.n_obj * 4
            ref_dirs = get_reference_directions("uniform", problem.n_obj, n_partitions=n_partitions)
            front = problem_.pareto_front(ref_dirs=ref_dirs)
        else:
            front = problem_.pareto_front()
    except Exception:
        pf_file_name = str(problem.name).upper() + "." + str(problem_.n_var) + "D.PF"
        front = np.loadtxt("./pf/" + pf_file_name)

    return front

def to_pymoo_problem(problem):
    problem_ = get_problem(problem.name, n_var=problem.n_var, n_obj=problem.n_obj)
    problem_.name = problem.name
    return problem_