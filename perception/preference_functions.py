import math

def get_preference_functions(function_names):
    preference_functions = {
        "abs_sine_function": lambda x: abs(math.sin(x/100 * math.pi/2)),
        "abs_cosine_function": lambda x: abs(math.cos(x/100 * math.pi/2))
    }

    return [preference_functions[function] for function in function_names]
