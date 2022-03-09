import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import load_model
from copy import deepcopy
import random

from evaluation.evaluation_functions import get_evaluation_functions
from movement.motor_system import execute_action
from movement.motor_system import opposite_direction
from perception.attentional_system import fixate_attention
from perception.preference_functions import get_preference_functions

class Environment:
    def __init__(self, terrain, agent, num_corrupted_perceptions=0, corruption_period=20, corruption_rate=0.0):
        self.terrain = terrain
        self.agent = agent

        self.position = terrain.get_middle_position()
        self.path = [self.position]
        self.last_direction = ""

        self.corrupted_perceptions = []
        self.num_corrupted_perceptions = num_corrupted_perceptions
        self.corruption_period = corruption_period
        self.corruption_rate = corruption_rate

    def update_path(self, position):
        self.path.append(position)

    def train_agent(self):
        evaluation_functions = get_evaluation_functions(self.agent.evaluation_functions)
        X_train = self.terrain.flatten_grid()

        for perception in range(self.agent.num_perceptions):
            eval_funct = evaluation_functions[perception]
            Y_train = np.apply_along_axis(eval_funct, 1, X_train)

            model = build_model(X_train.shape[1], 1, self.agent.hidden_layer_units)
            model.fit(X_train, Y_train, epochs=100, validation_split = 0.2, verbose=0)
            model.save("models/evaluation_function_" + str(perception) + ".h5")
            print("Model " + str(perception) + " saved successfully")

    def predict(self, positions, iteration):
        preference_functions = get_preference_functions(self.agent.preference_functions)
        predictions = {}

        for direction in ["U", "R", "D", "L"]:
            predictions[direction] = {}

        perceptions = [0, 1, 2, 3, 4]
        if iteration % self.corruption_period == 0:
            self.corrupted_perceptions = random.sample(perceptions, self.num_corrupted_perceptions)
            print("----------------------------------------")
            print("Corrupted perceptions = " + str(self.corrupted_perceptions))
            print("----------------------------------------")

        for perception in range(self.agent.num_perceptions):
            model = load_model("models/evaluation_function_" + str(perception) + ".h5")
            for direction in ["U", "R", "D", "L"]:
                magnitudes = positions[direction]
                if magnitudes.size > 0:
                    if perception in self.corrupted_perceptions:
                        corruption = abs(np.random.normal(0, self.corruption_rate*(iteration % self.corruption_period)))
                    else:
                        corruption = 0.0

                    X_test = np.array([magnitudes])
                    evaluation = np.squeeze(model.predict(X_test).flatten())
                    prediction = preference_functions[perception](evaluation + corruption)

                    predictions[direction][perception] = {
                        "magnitudes": magnitudes,
                        "prediction": prediction
                    }

        return predictions

    def action_error(self, action):
        evaluation_functions = get_evaluation_functions(self.agent.evaluation_functions)
        preference_functions = get_preference_functions(self.agent.preference_functions)

        perception = action["perception"]
        magnitudes = action["magnitudes"]
        prediction = action["module_predictions"]["evaluative_system"]
        evaluation = evaluation_functions[perception](magnitudes)
        real_value = preference_functions[perception](evaluation)

        evaluation_error = {
            "evaluation": real_value,
            "error": abs(real_value - prediction)
        }
        return evaluation_error


    def run_one_iteration(self, iteration, data):
        if not self.agent.is_dead():
            perceived_positions = self.terrain.perceive_environment(self.position)
            perceived_predictions = self.predict(perceived_positions, iteration)
            fixated_predictions = fixate_attention(perceived_predictions, self.agent.attentional_limit)

            action = self.agent.global_workspace.select_action(fixated_predictions, opposite_direction(self.last_direction),
                                                               self.agent.error_threshold, self.agent.error_activation_factor)
            self.position = execute_action(action["direction"], self.position)
            self.last_direction = action["direction"]
            self.update_path(self.position)
            evaluation_error = self.action_error(action)

            happiness = self.agent.happiness
            significance = action["significance"]
            evaluation = evaluation_error["evaluation"]
            error = evaluation_error["error"]
            direction = action["direction"]
            perception = action["perception"]
            evaluative_system_weights = self.agent.global_workspace.evaluative_system_weights
            memory_system_weights = self.agent.global_workspace.memory_system_weights
            evaluative_system_prediction = action["module_predictions"]["evaluative_system"]
            memory_system_error = action["module_predictions"]["memory_system"]
            memory = self.agent.global_workspace.memory.values
            corrupted_perceptions = self.corrupted_perceptions

            d = {
                "iteration": iteration + 1,
                "happiness": happiness,
                "significance": significance,
                "evaluation": evaluation,
                "error": error,
                "direction": direction,
                "perception": perception,
                "evaluative_system_weights": evaluative_system_weights,
                "memory_system_weights": memory_system_weights,
                "evaluative_system_prediction": evaluative_system_prediction,
                "memory_system_error": memory_system_error,
                "memory": memory,
                "corrupted_perceptions": corrupted_perceptions
            }
            data.append(deepcopy(d))

            print("> Iteration " + str(iteration + 1) + " | Happiness = " + str(happiness) + " | Position = " + str(self.position))
            print("Prediction = " + str(evaluative_system_prediction) + " | Real value = " + str(evaluation) + " | Error = " + str(error) + " | Perception = " + str(direction) + str(perception))
            print("Weights: \n\tES = " + str(evaluative_system_weights[perception]) + " * " + str(evaluative_system_prediction) +
                  "\n\tMS = " + str(memory_system_weights[perception]) + " * " + str(memory_system_error))

            self.agent.global_workspace.memory.add_value(perception, error)
            self.agent.global_workspace.update_weights(perception, error, self.agent.error_threshold)

            relative_error = error - self.agent.error_threshold
            if relative_error < 0.0:
                relative_error *= 5.0
            happiness_increase = (-self.agent.happiness_rate)*(relative_error)
            self.agent.happiness += happiness_increase
            if self.agent.happiness > self.agent.maximum_happiness:
                self.agent.happiness = self.agent.maximum_happiness

    def run_multiple_iterations(self, iterations):
        iterationsData = []
        for iteration in range(iterations):
            self.run_one_iteration(iteration, iterationsData)
            if self.agent.is_dead():
                print ("Death threshold reached in iteration " + str(iteration + 1))
                break

        data = {
            "terrain": {
                "grid_size": self.terrain.grid_size,
                "num_magnitudes": self.terrain.num_magnitudes,
                "max_value": self.terrain.max_value,
                "grid": self.terrain.grid.tolist()
            },
            "optimal_grid": self.optimal_grid().tolist(),
            "path": self.path,
            "iterations": iterationsData
        }
        return data

    def optimal_evaluation(self, X):
        evaluation_functions = get_evaluation_functions(self.agent.evaluation_functions)
        preference_functions = get_preference_functions(self.agent.preference_functions)
        return max(preference_functions[perception](evaluation_functions[perception](X)) for perception in range(self.agent.num_perceptions))

    def optimal_grid(self):
        return np.apply_along_axis(self.optimal_evaluation, 2, self.terrain.grid)


#######################
# Auxiliary functions #
#######################

def build_model(X_dim, Y_dim, hidden_layer_units):
    model = keras.Sequential([
        layers.Dense(X_dim, activation="relu", input_shape=[X_dim]),
        layers.Dense(hidden_layer_units, activation="relu"),
        layers.Dense(hidden_layer_units, activation="relu"),
        layers.Dense(Y_dim)
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(loss="mse",
                optimizer=optimizer,
                metrics=["mae", "mse"])

    return model
