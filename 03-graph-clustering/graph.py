import numpy as np
from TD.pointcloud import Point, Cloud

class Edge:
    """An edge in a Point graph.

    Attributes
    ----------
    p_1, p2 : int -- the vertices (indices of points) connected by the edge
    length : float -- the length of the edge
    """

    def __init__(self, p1, p2, length):
        self.p_1 = p1
        self.p_2 = p2
        self.length = length

    def __repr__(self):
        return f"Edge({self.p_1}, {self.p_2}, {self.length})"

    def __str__(self):
        return f"Edge: {self.p_1} -> {self.p_2} (length {self.length})"

    def __lt__(self,other) :
        return self.length < other.length
    


class Graph:
    """
    A simple weighted graph, with edges sorted by non-decreasing length.
    node_names are to indices of Points in an associated Cloud object.

    Attributes
    ----------
    edges : [Edge]
    node_names : [str]
    """

    def __init__(self):
        self.edges = []
        self.node_names = []

    def __str__(self):
        n = len(self.node_names)
        if n == 0:
            node_str = '0 Nodes'
        elif n == 1:
            node_str = f'1 Node: 0: {self.node_names[0]}'
        else:
            node_str = f'{n} Nodes: ' + ', '.join(
                f'{i}: {n}' for (i, n) in enumerate(self.node_names)
            )
        m = len(self.edges)
        if m == 0:
            edge_str = '0 Edges'
        elif m == 1:
            edge_str = f'1 Edge: {self.edges}'
        else:
            edge_str = f'{m} Edges: {self.edges}'
        return f'Graph:\n{node_str}\n{edge_str}'

    def __iter__(self):
        # Iterate over a Graph -> iterate over its edges list
        return iter(self.edges)

    def edge_count(self):
        return len(self.edges)

    def node_count(self):
        return len(self.node_names)

    def node_name(self, i: int) -> str:
        return self.node_names[i]

    def edge(self, i: int) -> Edge:
        """The i-th edge of the (length-sorted) edge list."""
        return self.edges[i]

    def add_nodes(self, ns: [str]) -> None:
        """Add a list of (names of) nodes to the graph."""
        self.node_names = self.node_names + ns


    def add_edges(self, es: [Edge]) -> None:
        """Add a list of edges to the graph,
        maintaining the length-sorted invariant.
        """
        self.edges = self.edges + es
        self.edges.sort()


def graph_from_cloud(c: Cloud):
    """Construct the complete graph whose nodes are names of points in c
    and where the length of the edge between two points is the Euclidean
    distance between them.
    """
    res = Graph()
    for (i,p) in enumerate(c) :
        res.add_nodes([p.name])
        res.add_edges([Edge(j,i,p.euclidean_distance(c[j])) for j in range(i)])
    
    return res


def graph_from_matrix(node_names: [str], dist_matrix: [[float]]):
    """Construct the complete graph on the given list of node names
    with the length of the edge between nodes i and j given by the
    (i,j)-th entry of the matrix.
    """
    n = len(node_names)
    res = Graph()
    res.add_nodes(node_names)
    for i in range(n) :
        for j in range(i) :
            new_edge = Edge(i,j,dist_matrix[i][j])
            res.add_edges([new_edge])
    return res

