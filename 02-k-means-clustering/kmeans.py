import numpy as np
import random

class Point:
    """A point in a dataset.

    Attributes:
        d: int             -- dimension of the ambient space
        coords: np.ndarray -- coordinates of the point
        label: int = 0     -- label of the cluster the point is assigned to
    """

    def __init__(self, d: int):
        assert d > 0
        self.d = d
        self.coords = np.zeros(d)
        self.label = 0

    def update_coords(self, new_coords: np.ndarray):
        assert len(new_coords) == self.d
        self.coords = np.array(new_coords, dtype = float).copy()

    def squared_dist(self, other) :
        diff = self.coords - other.coords
        return float(np.sum(diff*diff))



class Cloud:
    """A cloud of points for the k-means algorithm.

    Data attributes:
    - d: int              -- dimension of the ambient space
    - points: list[Point] -- list of points
    - k: int              -- number of centers
    - centers: np.ndarray -- array of centers
    """

    def __init__(self, d: int, k: int):
        self.d = d
        self.k = k
        self.points = []
        self.centers = np.array([Point(d) for _ in range(self.k)])

    def add_point(self, p: Point, label: int) -> None:
        """Copy p to the cloud, in cluster label."""
        new_point = Point(self.d)
        new_point.update_coords(p.coords)
        self.points.append(new_point)
        self.points[-1].label = label

    def intracluster_variance(self) -> float :
        n = len(self.points)
        res = 0
        for i in range(n) :
            res += self.points[i].squared_dist(self.centers[self.points[i].label])
        
        return res/n
    
    def set_voronoi_labels(self) -> int :
        res = 0

        n = len(self.points)
        for i in range(n) :
            closest_dist = float("inf")
            closest_label = -1
            for j in range(self.k) :
                dist = self.points[i].squared_dist(self.centers[j])
                if dist < closest_dist :
                    closest_dist = dist
                    closest_label = j
            if self.points[i].label != closest_label :
                res += 1
                self.points[i].label = closest_label
        
        return res

    def set_centroid_centers(self) -> None :
        n = len(self.points)
        k = self.k
        new_centers = np.array([Point(self.d) for _ in range(k)])

        for i in range(k) :
            new_centers[i].label = i

        numb_in_clust = np.array([0 for _ in range(k)])

        for i in range(n) :
            clust = self.points[i].label
            numb_in_clust[clust] += 1
            new_centers[clust].coords = np.add(new_centers[clust].coords, self.points[i].coords)
            
        for i in range(k) :
            if numb_in_clust[i] == 0 :
                continue
            else :
                self.centers[i].coords = np.copy(np.divide(new_centers[i].coords, numb_in_clust[i]))
        

    def init_random_partition(self) -> None :
        n = len(self.points)
        for i in range(n) :
            self.points[i].label = random.randrange(self.k)
        self.set_centroid_centers()

    def lloyd(self) -> None :
        """Lloyd’s algorithm.
        Assumes the clusters have already been initialized somehow.
        """
        while True:
            nb_changes = self.set_voronoi_labels()
            if nb_changes == 0:
                break
            self.set_centroid_centers()
    
    def init_forgy(self) -> None :
        """Forgy's initialization: distinct centers are sampled
        uniformly at random from the points of the cloud.
        """
        n = len(self.points)
        chosen_indices = random.sample(range(n), self.k)
        for i in range(self.k) :
            self.centers[i].update_coords(self.points[chosen_indices[i]].coords)
            self.centers[i].label = i
    
    def init_plusplus(self) -> None :
        n = len(self.points)
        
        first = random.randrange(n)
        self.centers[0].update_coords(self.points[first].coords)
        self.centers[0].label = 0

        min_dists = np.array(
            [self.points[i].squared_dist(self.centers[0]) for i in range(n)],
            dtype = float
        )

        chosen = {first}
        for c in range(1,self.k) :
            weights = min_dists.copy()
            for idx in chosen :
                weights[idx] = 0.0
            
            total = float(np.sum(weights))
            if total == 0.0 :
                remaining = [i for i in range(n) if i not in chosen]
                idx = random.choice(remaining)
            
            else :
                r = random.uniform(0.0,total)
                acc = 0.0
                idx = 0
                for i in range(n) :
                    acc += weights[i]
                    if r <= acc :
                        idx = i
                        break
                
            chosen.add(idx)
            self.centers[c].update_coords(self.points[idx].coords)
            self.centers[c].label = c
        
            for i in range(n) :
                d = self.points[i].squared_dist(self.centers[c])
                if d < min_dists[i] :
                    min_dists[i] = d