import numpy as np
from scipy.ndimage import gaussian_filter

class Terrain:
    def __init__(self, grid_size, num_magnitudes, max_value, gaussian_sigma):
        self.grid_size = grid_size
        self.num_magnitudes = num_magnitudes
        self.max_value = max_value
        self.gaussian_sigma = gaussian_sigma

        self.grid = np.random.randn(grid_size, grid_size, num_magnitudes)
        for i in range(num_magnitudes):
            self.grid[:, :, i] = (self.grid[:, :, i] - np.min(self.grid[:, :, i]))/np.ptp(self.grid[:, :, i])
            self.grid[:, :, i] = gaussian_filter(self.grid[:, :, i], sigma=gaussian_sigma)
            self.grid[:, :, i] = max_value*(self.grid[:, :, i]-self.grid[:, :, i].min())/np.ptp(self.grid[:, :, i])

    def get_middle_position(self):
        return [int(self.grid_size/2)] * 2

    def flatten_grid(self):
        return self.grid.reshape(self.grid.shape[0]*self.grid.shape[1], self.grid.shape[2])

    def perceive_environment(self, position):
        x = position[0]
        y = position[1]

        up = self.grid[x, y+1] if y+1 < self.grid.shape[1] else np.array([])
        right = self.grid[x+1, y] if x+1 < self.grid.shape[0] else np.array([])
        down = self.grid[x, y-1] if y > 0 else np.array([])
        left = self.grid[x-1, y] if x > 0 else np.array([])
        
        perceived_positions = {
            "U": up,
            "R": right,
            "D": down,
            "L": left
        }
        return perceived_positions
