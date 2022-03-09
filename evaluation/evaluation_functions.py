import numpy as np
import math

def ackley(x, d, a, b, c):
    sumSquares = 0.0
    sumCosines = 0.0
    for i in range(d):
        sumSquares = sumSquares + x[i]**2
        sumCosines = sumCosines + math.cos(c*x[i])

    return -(a*math.exp(-b*math.sqrt(sumSquares/d))) - math.exp(sumCosines/d) + a + math.exp(1)

def friedman(x):
    return 10.0*math.sin(math.pi*x[0]*x[1]) + 20.0*((x[2]-0.5)**2) + 10*x[3] + 5*x[4]

def select(x, n):
    return x[n]

def get_evaluation_functions(function_names, d = 5):
    evaluation_functions = {
        "sum_function": lambda x: np.sum(x),
        "distance_function": lambda x: np.linalg.norm(x),
        "ackley": lambda x: ackley(x, d, 20.0, 0.2, 2.0*math.pi),
        "modified_ackley": lambda x: ackley(x, d, 1.0, 2.0, 3.0),
        "friedman": lambda x: friedman(x),
        "select_0": lambda x: select(x, 0),
        "select_1": lambda x: select(x, 1),
        "select_2": lambda x: select(x, 2),
        "select_3": lambda x: select(x, 3),
        "select_4": lambda x: select(x, 4),
    }

    return [evaluation_functions[function] for function in function_names]
