import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.ndimage import gaussian_filter

def plot_data(data):
    terrain = data["terrain"]
    grid = np.array(terrain["grid"])
    path = np.array(data["path"])
    plot_terrains(grid, terrain["max_value"], terrain["num_magnitudes"], path)

    plot_iterations(data["iterations"])

    optimal_grid = np.array(data["optimal_grid"])
    plot_path(optimal_grid, path)

def plot_terrains(grid, max_value, num_magnitudes, path):
    cmap = plt.cm.get_cmap('jet', max_value)
    bounds = range(0, max_value + 1)
    norm = colors.BoundaryNorm(bounds, cmap.N)

    for i in range(num_magnitudes):
        fig, ax = plt.subplots()
        pos = ax.imshow(grid[:, :, i], cmap=cmap, norm=norm)
        ax.invert_yaxis()

        fig.colorbar(pos, ax=ax)

        plt.plot(*path.T, color="k")

        plt.xticks([0, 30, 60, 90, 120, 150])
        plt.yticks([0, 30, 60, 90, 120, 150])
        plt.show()

def plot_iterations(iterations):
    x = range(len(iterations))
    happiness = [dict["happiness"] for dict in iterations]
    error = [dict["error"] for dict in iterations]

    plt.plot(x, happiness, label = "felicidad")
    plt.plot(x, error, label = "error")
    
    plt.xlabel("Iteración")
    plt.ylabel("Felicidad/error")
    plt.title("Felicidad vs error")
    plt.legend()
    plt.show()

    prediction = [dict["evaluative_system_prediction"] for dict in iterations]
    real_value = [dict["evaluation"] for dict in iterations]

    plt.plot(x, prediction, label = "predicción", color="#FF7F0E", marker="o", mfc="none")
    plt.plot(x, real_value, label = "valor real", marker="s", mfc="none")
    
    plt.xlabel("Iteración")
    plt.ylabel("Predicción/valor")
    plt.title("Predicción vs valor real")
    plt.legend()
    plt.show()

    perception = [dict["perception"] for dict in iterations]
    corrupted_perceptions = np.array([dict["corrupted_perceptions"] for dict in iterations])

    xmax = range(1, len(iterations) + 1)
    fig, ax = plt.subplots(1, 1)
    for corrupted_perception in corrupted_perceptions.T:
        ax.hlines(corrupted_perception, x, xmax, color="red", linewidth=6)

    ax.hlines(perception, x, xmax, color="blue")
    plt.yticks([0, 1, 2, 3, 4])
    #plt.plot(x, perception, drawstyle="steps", color="blue")

    plt.xlabel("Iteración")
    plt.ylabel("Percepción")
    plt.title("Percepción elegida vs corrompidas")
    plt.show()

    evaluative_system_weights = np.array([dict["evaluative_system_weights"] for dict in iterations])
    i = 0
    for evaluative_system_weight in evaluative_system_weights.T:
        plt.plot(x, evaluative_system_weight, label="percepción " + str(i))
        i += 1

    plt.xlabel("Iteración")
    plt.ylabel("Peso")
    plt.title("Peso del sistema evaluativo")
    plt.legend()
    plt.show()

def plot_weights(iterations):
    x = range(len(iterations))
    happiness = [dict["happiness"] for dict in iterations]
    error = [dict["error"] for dict in iterations]
    perception = [dict["perception"] for dict in iterations]

    plt.plot(x, happiness, label = "happiness")
    plt.plot(x, error, label = "error")
    
    plt.xlabel("iterations")
    plt.ylabel("happiness/error")
    plt.title("Happiness and error.")
    plt.legend()

    plt.show()

    plt.plot(x, perception, label = "perception", drawstyle="steps")

    plt.xlabel("iterations")
    plt.ylabel("perceptions")
    plt.title("Perceptions.")
    plt.yticks([0, 1, 2, 3, 4])
    plt.legend()

    plt.show()


def plot_path(optimal_terrain, path):
    smooth_terrain = gaussian_filter(optimal_terrain[:, :], sigma=0.7)
    cmap = plt.cm.get_cmap('jet', 100)
    bounds = [x / 100 for x in range(100)]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots()
    pos = ax.imshow(smooth_terrain[:, :], cmap=cmap, norm=norm)
    ax.invert_yaxis()
    fig.colorbar(pos, ax=ax)

    plt.plot(*path.T, color="k")

    plt.xticks([0, 30, 60, 90, 120, 150])
    plt.yticks([0, 30, 60, 90, 120, 150])
    plt.show()
