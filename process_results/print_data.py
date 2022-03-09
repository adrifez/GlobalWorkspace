import numpy as np

def print_final_state(iterations, path):
    state = iterations[len(iterations)-1]
    print("\n==================================================")
    print("AGENT FINAL STATE")
    print("Happiness = " + str(state["happiness"]))
    print("Error = " + str(state["error"]))
    print("==================================================")
    print("EVALUATIVE SYSTEM WEIGHTS")
    i = 0
    for w in state["evaluative_system_weights"]:
        print("Perception " + str(i) + ": " + str(w))
        i += 1
    print("==================================================")
    print("MEMORY SYSTEM WEIGHTS")
    i = 0
    for w in state["memory_system_weights"]:
        print("Perception " + str(i) + ": " + str(w))
        i += 1
    print("==================================================")
    print("MEMORY")
    i = 0
    for values in state["memory"]:
        print("Perception " + str(i) + ": " + str(values))
        i += 1
    print("==================================================")
    print("PATH")
    for position in path:
        print(str(position) + " -> ", end = '')
    print("END")
    print("==================================================")
