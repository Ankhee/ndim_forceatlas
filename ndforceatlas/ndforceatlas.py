"""
Implementation of Gephi's ForceAtlas2 algorithm, adapted to work in specified
n-dimensional space.


ndforceatlas.ndforceatlas(g) takes a numpy.ndarray nodes adjacency array, 
and returns list of (ndarray, ...) tuples. 

Alternatively, ndforceatlas.ndforceatlas_networkx(g) takes in a networkx graph.
 
For more info on ForceAtlas2, see:

Jacomy M, Venturini T, Heymann S, Bastian M (2014) ForceAtlas2, 
a Continuous Graph Layout Algorithm for Handy Network Visualization Designed 
for the Gephi Software. PLoS ONE 9(6): e98679. 
https://doi.org/10.1371/journal.pone.0098679
"""

import math
import statistics
import numpy as np
from ndforceatlas import ndfa_utils


def ndforceatlas(
        g: np.ndarray,
        n_dim=2,
        n_iter=512,

        scaling=2.0,
        gravity=1.0,
        use_strong_gravity=False,

        nodes_barnes_hut_optimization=True,
        edges_barnes_hut_optimization=False,
        barnes_hut_theta=1.2,

        edge_weight_influence=1,
        dissuade_hubs=False,
        linlog_mode=False,

        periodic_autoscale=True,
        verbose=False
):
    # assertions go here
    assert isinstance(g, np.ndarray), \
        "Adjacency matrix is not a numpy ndarray"
    assert g.shape == (g.shape[0], g.shape[0]), \
        "Adjacency matrix is not a 2D square"
    if periodic_autoscale & n_iter < 64:
        periodic_autoscale = False

    # initialize
    # ===========

    if verbose:
        print("[N_DIM_FORCEATLAS] Initializing nodes...")

    init_range = g.shape[0]
    nodes = []
    for i in range(g.shape[0]):
        node = ndfa_utils.Node()
        node.pos = np.random.random(n_dim) * init_range - init_range * 0.5
        node.force = np.zeros(n_dim)
        node.old_force = np.zeros(n_dim)
        node.mass = np.sum(g[i])  # sum of all edge weights for given node
        nodes.append(node)

    if verbose:
        print("[N_DIM_FORCEATLAS] Initializing edges...")

    edges = []
    edges_array = np.asarray(g.nonzero()).T
    for e in edges_array:
        if e[0] <= e[1]:
            continue
        edge = ndfa_utils.Edge()
        edge.n1 = nodes[e[0]]
        edge.n2 = nodes[e[1]]
        edge.weight = g[tuple(e)]
        edges.append(edge)

    if periodic_autoscale:
        sequence = []
        if n_iter > 128:
            autoscale_iter_threshold = 16
        else:
            autoscale_iter_threshold = math.floor(n_iter / 4)

    # iterate
    # ===========

    if verbose:
        print("[N_DIM_FORCEATLAS] Initializing simulation...")

    for i in range(n_iter):

        ndfa_utils.graph_apply_attraction(
            _edges=edges,
            _nodes=nodes,
            _edge_weight_influence=edge_weight_influence,
            _dissuade_hubs=dissuade_hubs,
            _use_barnes_hut=edges_barnes_hut_optimization,
            _barnes_hut_theta=barnes_hut_theta,
            _linlog_mode=linlog_mode)

        ndfa_utils.graph_apply_repulsion(
            _nodes=nodes,
            _use_barnes_hut=nodes_barnes_hut_optimization,
            _barnes_hut_theta=barnes_hut_theta,
            _scaling=scaling)

        ndfa_utils.graph_apply_gravity(
            _nodes=nodes,
            _gravity=gravity,
            _use_strong_gravity=use_strong_gravity)

        ndfa_utils.graph_calculate_swing(
            _nodes=nodes)

        ndfa_utils.graph_resolve_forces(
            _nodes=nodes)

        if periodic_autoscale:
            mean = statistics.mean([np.asscalar(np.linalg.norm(node.pos))
                                    for node in nodes])
            sequence.append(mean)
            if (i + 1) % autoscale_iter_threshold == 0:
                ndfa_utils.graph_resolve_geometric_progression_autoscale(
                    sequence, nodes)
                sequence = []

        if verbose and i % 32 == 0:
            print("[N_DIM_FORCEATLAS] "
                  "Progress: %.3f%%" % (i / n_iter * 100))

    if verbose:
        print("[N_DIM_FORCEATLAS] Simulation complete.")

    return ndfa_utils.output(nodes)


def ndforceatlas_networkx(g, **kwargs):
    import networkx
    assert isinstance(g, networkx.classes.graph.Graph), \
        "Graph is not a NetworkX graph."

    m = networkx.to_numpy_matrix(g)
    ndforceatlas(m, **kwargs)
