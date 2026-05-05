"""
A basic dendrogram for single-linkage hierarchical clustering.
"""

from TD.graph import Graph, Edge


class Dendrogram:
    """
    Attributes
    ----------

    g : Graph -- the underlying graph
    n : int -- the number of nodes in the underlying graph
    parent : [int] -- parents for union-find
    rank : [int] -- ranks for union-find
    left : [int] -- binary tree lefts
    down : [int] -- binary tree downs
    height : [float]
        heights at which represented cluster is merged into parent
    cut_height : float -- current cut height
    cluster : [int] -- list of cluster representatives
    total_clusters : int -- number of clusters currently identified
    ns_clusters : int -- number of non-singleton clusters
    significant_heights : [float] -- List of significant heights (Ex. 5)
    """

    def __init__(self, g: Graph):
        self.g = g
        self.n = g.node_count()

        self.parent = [-1] * self.n
        self.rank = [0] * self.n
        self.left = [-1] * self.n
        self.down = [-1] * self.n
        self.height = [-1] * self.n
        self.cluster = [-1] * self.n

        self.cut_height = -1
        self.total_clusters = 0
        self.ns_clusters = 0
        self.significant_heights = []

    def __str__(self):
        lines = ["node\tparent\trank\tleft\tdown\theight\tcluster"]
        for i, vals in enumerate(
            zip(self.parent, self.rank, self.left, self.down, self.height, self.cluster)
        ):
            s = f"{i}\t"
            s += "\t".join("-" if v == -1 else str(v) for v in vals)
            lines.append(s)
        return "\n".join(lines)

    def get_dendrogram_height(self) -> float:
        """Height of the dendrogram."""
        return self.height[self.down[self.representative(0)]]

    def representative(self, i: int) -> int:
        """Find the representative in the union-find structure
        for the cluster containing node i.
        """
        assert 0 <= i < self.n
        result = i
        while self.parent[result] != -1 :
            result = self.parent[result]
        return result

    def merge(self, e: Edge):
        """Merge the clusters connected by the edge."""
        # TODO: Exercise 7
        # Plan:
        # 1. Find the representatives
        i,j = e.p_1, e.p_2
        rep_i = self.representative(i)
        rep_j = self.representative(j)
        # 2. Choose the highest
        if rep_i == rep_j :
            return
        if self.rank[rep_i] > self.rank[rep_j] :
            self.parent[rep_j] = rep_i
            self.left[rep_j] = self.down[rep_i]
            self.down[rep_i] = rep_j
            for ed in self.g.edges :
                if ed.p_1 == rep_j or ed.p_2 == rep_j :
                    self.height[rep_j] = ed.length/2
                    break

        elif self.rank[rep_j] > self.rank[rep_i] :
            self.parent[rep_i] = rep_j
            self.left[rep_i] = self.down[rep_j]
            self.down[rep_j] = rep_i
            for ed in self.g.edges :
                if ed.p_1 == rep_j or ed.p_2 == rep_j :
                    self.height[rep_i] = ed.length/2
                    break

        elif rep_i > rep_j :
            self.parent[rep_j] = rep_i
            self.rank[rep_i] += 1
            self.left[rep_j] = self.down[rep_i]
            self.down[rep_i] = rep_j
            for ed in self.g.edges :
                if ed.p_1 == rep_j or ed.p_2 == rep_j :
                    self.height[rep_j] = ed.length/2
                    break
            
        else :
            self.parent[rep_i] = rep_j
            self.rank[rep_j] += 1
            self.left[rep_i] = self.down[rep_j]
            self.down[rep_j] = rep_i
            for ed in self.g.edges :
                if ed.p_1 == rep_i or ed.p_2 == rep_i :
                    self.height[rep_i] = ed.length/2
                    break

        # 3. Adjust parent, left, and down
        # 4. Update ranks
        # 5. Update heights
        pass

    def build(self):
        """Merge along each edge in non-decreasing length order
        to build the dendrogram.
        """
        self.g.edges.sort()
        for ed in self.g.edges :
            self.merge(ed)

    def find_heights(self, eps: float):
        """Put all heights <= eps into list of significant heights."""
        assert eps != 0.0
        h: float = self.get_dendrogram_height()
        slots = int(1 / eps + 1)
        buckets = [0.0] * slots
        for i, h_i in enumerate(self.height):
            if h_i == -1:
                continue
            q = int(h_i / eps / h)
            if h_i > buckets[q]:
                buckets[q] = h_i
        self.significant_heights = [x for x in buckets if x > 0]

    def _set_clusters(self, i: int, h: float):
        assert 0 <= i < self.n
        if self.parent[i] != i and self.height[i] <= h :
            self.cluster[i] = self.cluster[self.parent[i]]
        else :
            self.cluster[i] = i
            self.total_clusters += 1
        self._set_clusters(self.left[i],h)
        self._set_clusters(self.down[i],h)

    def set_clusters(self, h: float):
        """(Re)set clusters with cut height h."""
        if self.cut_height is h:
            return  # Already done!  Do nothing.
        self.cut_height = h
        self._set_clusters(self.representative(0), h)

    def _count_ns_clusters(self):
        """Count non-singleton clusters from scratch"""
        count = 0
        check = [0]*self.n
        for i in range(self.n) :
            check[self.cluster[i]] += 1
        for i in range(self.n) :
            if check[i] > 1 :
                count += 1
        return count

    def count_ns_clusters(self) -> int:
        """The number of non-singleton clusters."""
        if self.ns_clusters == 0:
            self.ns_clusters = self._count_ns_clusters()
        return self.ns_clusters

    def clear_clusters(self):
        """Discard clustering data."""
        for i in range(len(self.cluster)):
            self.cluster[i] = -1
        self.ns_clusters = 0
        self.total_clusters = 0
        self.cut_height = -1

    def get_cluster_height(self, cluster: int) -> float:
        """Compute the height of the cluster c.
        Assumes build() has been called and the clusters of height <= h
        have been computed using set_clusters(h).
        """
        assert 0 <= cluster < self.n
        res = 0
        for i in range(self.n) :
            if self.cluster[i] == cluster :
                res = max(res,self.height[i])
        return res

        return 0  # Unreachable!

    def node_name(self, i: int) -> str:
        """The name of the i-th node in the underlying graph."""
        assert 0 <= i < self.n
        return self.g.node_name(i)

    # // For testing only

    def print_node(self, i: int):
        """For testing: print the i-th node."""
        print(
            f"Node (parent = {self.parent[i]},"
            f" down = {self.down[i]},"
            f" left = {self.left[i]},"
            f" rank = {self.rank[i]},"
            f" height = {self.height[i]},"
            f" cluster = {self.cluster[i]})"
        )

    def print_clusters(self):
        """For testing: print all clusters."""
        print_after = [-1] * self.n
        for i, c_i in enumerate(self.cluster):
            if c_i != i:
                print_after[i] = print_after[c_i]
                print_after[c_i] = i
        for i, c_i in enumerate(self.cluster):
            if c_i == i:  # Cluster rep
                print(
                    f'Cluster "{c_i}" (node: {i}; height: {self.get_cluster_height(i)})'
                )
                next_c = print_after[i]
                while next_c != -1:
                    print(self.node_name(next_c))
                    next_c = print_after[next_c]
