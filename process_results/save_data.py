import sys
import json

DATA_FILE = "results/exp_data.json"

def save_data(data, execution):
    DATA_FILE = "results/exp" + str(execution) + "_data.json"
    try:
        data_file = open(DATA_FILE, "w")
    except OSError:
        print("Could not open/read file:", DATA_FILE)
        sys.exit()

    data_file.write(json.dumps(data, indent=4))
    data_file.close()

def load_data(execution):
    DATA_FILE = "results/exp" + str(execution) + "_data.json"
    try:
        data_file = open(DATA_FILE, "r")
    except OSError:
        print("Could not open/read file:", DATA_FILE)
        sys.exit()

    data = json.load(data_file)
    data_file.close()
    return data

def save_terrain(terrain):
    TERRAIN_FILE = "results/random_terrain.json"
    try:
        terrain_file = open(TERRAIN_FILE, "w")
    except OSError:
        print("Could not open/read file:", TERRAIN_FILE)
        sys.exit()

    terrainDict = {
        "grid_size": terrain.grid_size,
        "num_magnitudes": terrain.num_magnitudes,
        "max_value": terrain.max_value,
        "grid": terrain.grid.tolist()
    }
    terrain_file.write(json.dumps(terrainDict, indent=4))
    terrain_file.close()
