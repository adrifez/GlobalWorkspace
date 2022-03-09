import sys
import json
import numpy as np

from environment import Environment
from terrain import Terrain
from agent import Agent
from process_results.print_data import print_final_state
from process_results.save_data import load_data
from process_results.save_data import save_data
from process_results.save_data import save_terrain
from process_results.plot_data import plot_data
from process_results.plot_data import plot_path

PROPERTIES_FILE = "properties.json"

if __name__ == "__main__":
    # Handle arguments
    if len(sys.argv) < 2:
        mode = "all"
    else:
        mode = sys.argv[1]

    # Load experiment properties
    try:
        properties_file = open(PROPERTIES_FILE, "r")
    except OSError:
        print("Could not open/read file:", PROPERTIES_FILE)
        sys.exit()

    properties = json.load(properties_file)
    properties_file.close()

    terrain_properties = properties["terrain_properties"]
    agent_properties = properties["agent_properties"]
    environment_properties = properties["environment_properties"]

    agent = Agent(agent_properties["num_perceptions"],
                  agent_properties["hidden_layer_units"],
                  agent_properties["evaluation_functions"],
                  agent_properties["preference_functions"],
                  agent_properties["attentional_limit"],
                  agent_properties["adaptational_rate"],
                  agent_properties["maximum_happiness"],
                  agent_properties["happiness_rate"],
                  agent_properties["error_threshold"],
                  agent_properties["error_activation_factor"],
                  agent_properties["happiness_thresholds"])
    

    if mode == "train" or mode == "all":
        # Train agent
        train_terrain = Terrain(terrain_properties["grid_size"],
                                terrain_properties["num_magnitudes"],
                                terrain_properties["max_value"],
                                terrain_properties["gaussian_sigma"])

        train_environment = Environment(train_terrain, agent)
        train_environment.train_agent()

    if mode == "test" or mode == "all":
        #Test agent
        test_terrain = Terrain(terrain_properties["grid_size"],
                               terrain_properties["num_magnitudes"],
                               terrain_properties["max_value"],
                               terrain_properties["gaussian_sigma"])

        test_environment = Environment(test_terrain,
                                       agent,
                                       environment_properties["num_corrupted_perceptions"],
                                       environment_properties["corruption_period"],
                                       environment_properties["corruption_rate"])

        print("\n##################")
        print("# STARTING AGENT #")
        print("##################\n")
        data = test_environment.run_multiple_iterations(environment_properties["iterations"])

        print_final_state(data["iterations"], data["path"])
        save_data(data, properties["execution"])

        # Update execution in experiment properties
        try:
            properties_file = open(PROPERTIES_FILE, "w")
        except OSError:
            print("Could not open/read file:", PROPERTIES_FILE)
            sys.exit()

        properties["execution"] += 1
        properties_file.write(json.dumps(properties, indent=4))
        properties_file.close()

    if mode == "plot" or mode == "all":
        if len(sys.argv) > 2:
            execution = sys.argv[2]
        else:
            execution = properties["execution"]
        data = load_data(execution)
        plot_data(data)
