import math
import statistics

import numpy as np

from ndforceatlas import ndbarneshut


class Node:
    def __init__(self):
        self.pos = np.zeros(1)
        self.force = np.zeros(1)
        self.old_force = np.zeros(1)
        self.size = 0.0
        self.mass = 0.0
        self.swing = 0.0
        self.speed = 1.0
        self.old_speed = 1.0

    def __repr__(self):
        return "<Node object: Pos: %s, Size: %.3f, Mass: %.3f>" % (
            self.pos, self.size, self.mass
        )

    def scale(self, _scale_factor):
        self.pos *= _scale_factor


class Edge:
    def __init__(self):
        self.n1 = Node()
        self.n2 = Node()
        self.weight = 0

    def __repr__(self):
        return "<Edge object: n1: %s, n2: %s, weight: %.3f>" % (
            self.n1, self.n2, self.weight
        )


def apply_attraction(
        _n1,
        _n2,
        _edge_weight_influence=1.0,
        _scaling=1.0,
        _dissuade_hubs=False,
        _linlog_mode=False):
    assert isinstance(_n1, Node), "_n1 has to be of class Node."
    if isinstance(_n2, Node):
        dist = np.linalg.norm(_n2.pos - _n1.pos)
        if _linlog_mode:
            dist = math.log(dist + 1)
        force = _edge_weight_influence * dist
        n1_to_n2 = (_n2.pos - _n1.pos) / dist * force * _scaling

        if _dissuade_hubs:
            _n1.force += n1_to_n2 / (_n1.mass + 1)
            _n2.force -= n1_to_n2 / (_n2.mass + 1)
        else:
            _n1.force += n1_to_n2
            _n2.force -= n1_to_n2  # n2_to_n1 = -(n1_to_n2)

    elif isinstance(_n2, np.ndarray):  # fake barnes-hut node
        dist = np.linalg.norm(_n2 - _n1.pos)
        force = _edge_weight_influence * dist
        n1_to_n2 = (_n2 - _n1.pos) / dist * force * _scaling

        if _dissuade_hubs:
            _n1.force += n1_to_n2 / (_n1.mass + 1)
        else:
            _n1.force += n1_to_n2

    else:
        raise AssertionError("_n2 has to be a numpy.ndarray or a Node.")


def apply_repulsion(
        _n1,
        _n2,
        _scaling=1.0):
    assert isinstance(_n1, Node), "_n1 has to be of class Node."
    if isinstance(_n2, Node):
        dist = np.linalg.norm(_n2.pos - _n1.pos)
        force = _scaling * (_n1.mass + 1) * (_n2.mass + 1) / dist
        n1_to_n2 = (_n2.pos - _n1.pos) / dist * force

        _n1.force -= n1_to_n2
        _n2.force += n1_to_n2  # n2_to_n1 = -(n1_to_n2)

    elif isinstance(_n2, tuple):  # fake barnes-hut node
        dist = np.linalg.norm(_n2[0] - _n1.pos)
        force = _scaling * (_n1.mass + 1) * (_n2[1] + 1) / dist
        n1_to_n2 = (_n2[0] - _n1.pos) / dist * force

        _n1.force -= n1_to_n2

    else:
        raise AssertionError("_n2 has to be a numpy.ndarray or a Node.")


def apply_gravity(
        _n,
        _gravity=1.0,
        _use_strong_gravity=False):
    force = _gravity * (_n.mass + 1)
    dist = np.linalg.norm(_n.pos)
    if _use_strong_gravity:
        force *= dist

    translation = -_n.pos / dist * force
    _n.force += translation


def graph_apply_attraction(
        _edges,
        _nodes,
        _edge_weight_influence=1,
        _use_barnes_hut=False,
        _barnes_hut_theta=1.2,
        **kwargs):
    if _use_barnes_hut:
        n_dim = len(_nodes[0].pos)
        length = max([np.linalg.norm(node.pos)] for node in _nodes)[0] * 4
        for node in _nodes:
            node_edges = [edge for edge in _edges
                          if (edge.n1 == node or edge.n2 == node)]
            found_nodes = [(edge.n2, edge.weight) for edge in node_edges]
            bhtree = ndbarneshut.Node(np.zeros(n_dim), length)
            bhtree.fit(found_nodes)
            bhtree.calculate_coms()
            neighbors = bhtree.neighbors(
                (node.pos, node.mass),
                _barnes_hut_theta)

            if _edge_weight_influence == 0:
                for neighbor in neighbors:
                    apply_attraction(
                        _n1=node,
                        _n2=neighbor[0],
                        _edge_weight_influence=1,
                        _scaling=0.5,
                        **kwargs)
            elif _edge_weight_influence == 1:
                for neighbor in neighbors:
                    apply_attraction(
                        _n1=node,
                        _n2=neighbor[0],
                        _edge_weight_influence=neighbor[1],
                        _scaling=0.5,
                        **kwargs)
            else:
                for neighbor in neighbors:
                    apply_attraction(
                        _n1=node,
                        _n2=neighbor[0],
                        _edge_weight_influence=pow(
                            neighbor[1], _edge_weight_influence),
                        _scaling=0.5,
                        **kwargs)
    else:
        if _edge_weight_influence == 0:
            for edge in _edges:
                apply_attraction(
                    _n1=edge.n1,
                    _n2=edge.n2,
                    _edge_weight_influence=1,
                    **kwargs)
        elif _edge_weight_influence == 1:
            for edge in _edges:
                apply_attraction(
                    _n1=edge.n1,
                    _n2=edge.n2,
                    _edge_weight_influence=edge.weight,
                    **kwargs)
        else:
            for edge in _edges:
                apply_attraction(
                    _n1=edge.n1,
                    _n2=edge.n2,
                    _edge_weight_influence=pow(
                        edge.weight, _edge_weight_influence),
                    **kwargs)


def graph_apply_repulsion(
        _nodes,
        _use_barnes_hut=True,
        _barnes_hut_theta=1.2,
        **kwargs):
    if _use_barnes_hut:
        n_dim = len(_nodes[0].pos)
        length = max([np.linalg.norm(node.pos)] for node in _nodes)[0] * 4
        tree_nodes = [(node.pos, node.mass) for node in _nodes]
        bhtree = ndbarneshut.Node(np.zeros(n_dim), length)
        bhtree.fit(tree_nodes)
        bhtree.calculate_coms()

        for node in _nodes:
            neighbors = bhtree.neighbors((node.pos, node.mass),
                                         theta=_barnes_hut_theta)
            for neighbor in neighbors:
                apply_repulsion(
                    _n1=node,
                    _n2=(neighbor[0], neighbor[1]),
                    **kwargs)

    else:
        for i in range(len(_nodes)):
            for j in range(i):
                apply_repulsion(_nodes[i], _nodes[j], **kwargs)


def graph_apply_gravity(
        _nodes,
        **kwargs):
    for node in _nodes:
        apply_gravity(node, **kwargs)


def graph_calculate_swing(
        _nodes):
    node_speed_const = 0.1
    node_speed_max_const = 100
    tau = 1
    for node in _nodes:
        node.swing = np.linalg.norm(node.force - node.old_force)
    graph_swing = sum([(node.mass + 1) * node.swing for node in _nodes])
    graph_traction = sum(
        [np.linalg.norm(node.force + node.old_force) * 0.5 * (node.mass + 1)
         for node in _nodes])
    graph_speed = tau * graph_traction / graph_swing
    for node in _nodes:
        node.speed = (node_speed_const
                      * graph_speed
                      / (1 + graph_speed) * math.sqrt(node.swing))
        node.speed = min(node.speed,
                         node_speed_max_const / np.linalg.norm(node.force))
        node.speed = min(node.speed, node.old_speed * 1.5)


def graph_resolve_forces(_nodes):
    for node in _nodes:
        # print(node.speed)
        node.pos += node.force * node.speed
        node.old_force = node.force
        node.force = np.zeros(len(node.pos))
        node.old_speed = node.speed


def graph_resolve_geometric_progression_autoscale(_sequence, _nodes):
    """Geometric progression formula: a(n) = a*r**(n-1)
    We're finding that formula, and 'jumping' some steps forward
    to converge faster.
    Then we're going to scale current nodes positions 
    to found 'future' positions."""
    dist = math.floor(len(_sequence) / 2)
    first_half = _sequence[:dist]
    second_half = _sequence[dist:]
    first_mean = statistics.mean(first_half)
    second_mean = statistics.mean(second_half)
    r = (second_mean / first_mean) ** (1 / dist)
    a = first_mean / r

    # the less change was observed, the more we can fast-forward
    if abs(r - 1) < 0.001:
        jump_factor = 32
    elif abs(r - 1) < 0.01:
        jump_factor = 16
    elif abs(r - 1) < 0.05:
        jump_factor = 8
    else:
        jump_factor = 4

    current_length = statistics.mean([np.asscalar(np.linalg.norm(node.pos))
                                      for node in _nodes])
    target_length = a * (r ** jump_factor)
    scale_factor = target_length / current_length

    for node in _nodes:
        node.scale(scale_factor)


def output(_nodes):
    return [node.pos for node in _nodes]
