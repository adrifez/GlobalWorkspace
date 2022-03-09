import numpy as np

class ShortTermMemory:
    MEMORY_CAPACITY = 1

    def __init__(self, num_perceptions):
        self.values = [[] for i in range(num_perceptions)]
        self.next_spaces = [0] * num_perceptions

    def is_empty(self, perception):
        return not self.values[perception]

    def is_full(self, perception):
        return len(self.values[perception]) == ShortTermMemory.MEMORY_CAPACITY

    def add_value(self, perception, value):
        if self.is_full(perception):
            space = self.next_spaces[perception]
            self.values[perception][space] = value
            self.next_spaces[perception] = (space + 1) % ShortTermMemory.MEMORY_CAPACITY
        else:
            self.values[perception].append(value)

    def get_perception_mean(self, perception):
        if self.is_empty(perception):
            return 0.0
        else:
            return np.mean(self.values[perception])

    def get_significance(self, perception):
        return self.get_perception_mean(perception)
