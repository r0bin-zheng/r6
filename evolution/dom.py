
# This file computes dominance relation between two evaluates solutions
# ind1 dominates ind2, output 1
# ind2 dominates ind1, output 2
# otherwise, ouput 0


# 评估两个点之间的优势关系
# ind1.fitness.wvalues是加权后的适配度
# judge the Pareto dominance relation between two evaluated solutions
def pareto_dominance(ind1, ind2):
    if ind1.fitness.valid and ind2.fitness.valid:
        r = get_pareto_dom_rel(ind1.fitness.wvalues, ind2.fitness.wvalues)
        return get_inverted_dom_rel(r)
    else:
        raise TypeError("Pareto dominance comparison cannot be done "
                        "when either of two individuals has not been evaluated")


# 评估两个点之间的θ优势关系
# judge the scalar (theta) dominance relation between two evaluated solutions
def scalar_dominance(ind1, ind2):
    if ind1.fitness.valid and ind2.fitness.valid:
        if ind1.cluster_id != ind2.cluster_id:
            return 0
        else:
            if ind1.scalar_dist < ind2.scalar_dist:
                return 1
            else:
                return 2
    else:
        raise TypeError("Scalar dominance comparison cannot be done "
                        "when either of two individuals has not been evaluated")


# TODO：添加其他优势关系
def epsilon_dominance(ind1, ind2, epsilon=0.1):
    if ind1.fitness.valid and ind2.fitness.valid:
        r = get_percentage_epsilon_dom_rel(ind1.fitness.wvalues, ind2.fitness.wvalues, epsilon)
        return get_inverted_dom_rel(r)
    else:
        raise TypeError("Epsilon dominance comparison cannot be done "
                        "when either of two individuals has not been evaluated")


# 评估两个点之间的优势关系
# 0 表示不可比，1 表示 ind1 优于 ind2，2 表示 ind2 优于 ind1
def get_pareto_dom_rel(values1, values2):
    n1, n2 = 0, 0
    for v1, v2 in zip(values1, values2):
        if v1 < v2:
            n1 += 1
        elif v2 < v1:
            n2 += 1

        if n1 > 0 and n2 > 0:
            return 0

    if n2 == 0 and n1 > 0:
        return 1
    elif n1 == 0 and n2 > 0:
        return 2
    else:
        return 0
    

def get_epsilon_dom_rel(values1, values2, epsilon=0.1):
    n1, n2 = 0, 0
    for v1, v2 in zip(values1, values2):
        if v1 < v2 - epsilon:
            n1 += 1
        elif v2 < v1 - epsilon:
            n2 += 1

        if n1 > 0 and n2 > 0:
            return 0

    if n2 == 0 and n1 > 0:
        return 1
    elif n1 == 0 and n2 > 0:
        return 2
    else:
        return 0


def get_percentage_epsilon_dom_rel(values1, values2, epsilon_percentage=0.1):
    n1, n2 = 0, 0
    for v1, v2 in zip(values1, values2):
        epsilon = epsilon_percentage * max(abs(v1), abs(v2))
        if v1 < v2 - epsilon:
            n1 += 1
        elif v2 < v1 - epsilon:
            n2 += 1

        if n1 > 0 and n2 > 0:
            return 0

    if n2 == 0 and n1 > 0:
        return 1
    elif n1 == 0 and n2 > 0:
        return 2
    else:
        return 0


# 将优势关系转换为相反的优势关系
def get_inverted_dom_rel(r):
    return r if r == 0 else 3 - r


def access_dom_rel(i, j, archive, rel_map, dom):
    if rel_map[i, j] != -1:
        # 如果是已经计算过的优势关系，直接返回
        return rel_map[i, j]
    else:
        # 否则计算优势关系
        r = dom(archive[i], archive[j])
        rel_map[i, j] = r
        rel_map[j, i] = get_inverted_dom_rel(r)
        return r
