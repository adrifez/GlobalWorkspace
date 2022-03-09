import math

from memories.short_term_memory import ShortTermMemory

class GlobalWorkspace:    
    def __init__(self, num_perceptions, adaptational_rate):
        self.evaluative_system_weights = [1/2] * num_perceptions
        self.memory_system_weights = [1/2] * num_perceptions

        self.memory = ShortTermMemory(num_perceptions)

        self.adaptational_rate = adaptational_rate
    
    def select_action(self, predictions, forbidden_direction, error_threshold, error_activation_factor = 1.0):
        selected_action = {}
        selected_action_value = -5.0

        for direction_perception, action in predictions.items():
            direction = direction_perception[0]
            if direction != forbidden_direction:
                perception = int(direction_perception[1:])
                magnitudes = action["magnitudes"]

                evaluative_system_weight = self.evaluative_system_weights[perception]
                memory_system_weight = self.memory_system_weights[perception]

                evaluative_system_value = action["prediction"]
                memory_system_value = self.memory.get_significance(perception)

                significance = (0.2*evaluative_system_weight*evaluative_system_value -
                                memory_system_weight*math.tanh(error_activation_factor*(memory_system_value - error_threshold)))

                if significance > selected_action_value:
                    selected_action_value = significance
                    selected_action["direction"] = direction
                    selected_action["perception"] = perception
                    selected_action["magnitudes"] = magnitudes
                    selected_action["significance"] = significance
                    selected_action["module_predictions"] = {
                        "evaluative_system": evaluative_system_value,
                        "memory_system": memory_system_value
                    }

        return selected_action

    def update_weights(self, perception, error, error_threshold):
        evaluative_system_weight, memory_system_weight = self.get_new_evaluative_memory_weights(perception, error, error_threshold)

        self.evaluative_system_weights[perception] = evaluative_system_weight
        self.memory_system_weights[perception] = memory_system_weight


    def get_new_evaluative_memory_weights(self, perception, error, error_threshold):
        if error <= error_threshold:
            relative_error = (error_threshold - error) / error_threshold
            evaluative_system_weight = self.evaluative_system_weights[perception]
            evaluative_system_weight += self.adaptational_rate*relative_error
            if evaluative_system_weight > 1.0:
                evaluative_system_weight = 1.0
            memory_system_weight = 1.0 - evaluative_system_weight
        else:
            relative_error = (error - error_threshold) / (1 - error_threshold)
            memory_system_weight = self.memory_system_weights[perception]
            memory_system_weight += self.adaptational_rate*relative_error
            if memory_system_weight > 1.0:
                memory_system_weight = 1.0
            evaluative_system_weight = 1.0 - memory_system_weight

        return evaluative_system_weight, memory_system_weight
