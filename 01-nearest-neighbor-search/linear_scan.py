"""
linear_scan module
"""
import numpy as np
from TD.nearest_neighbor import NearestNeighborSearch


class LinearScan(NearestNeighborSearch):
    def query(self, x):
        # Ensures x is of correct shape
        super().query(x)
        # Store the index of nearest neighbor
        nearest_neighbor_index = -1
        current_min_dist = np.inf

        for ind,point in enumerate(self.X) :
            dist = self.metric(x, point)
            if dist < current_min_dist :
                current_min_dist = dist
                nearest_neighbor_index = ind
        
        
        return current_min_dist, nearest_neighbor_index
