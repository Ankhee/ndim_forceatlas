"""Implementation of Barnes-Hut tree algorithm (quadtree, octatree...) in n-dimensions.

Barnes-Hut algorithm is an approximation algorithm for performing an n-body simulation.
BHTree generation recursively divides n-dim space into cells, which contain 0 or 1 bodies.
This algorithm is used to approximate forces acting on a body. Group of bodies sufficently away
from queried body can be approximated to one center of mass.

See:
    https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation
    http://arborjs.org/docs/barnes-hut
    
See example implementation:
    https://codereview.stackexchange.com/questions/43749/
    
Terminology:
    1.  Node            - Basic element of BHTree structure. 
                          Sector can be either:
        a.  Empty       - Doesn't contain any bodies,
        b.  External    - Contains precisely one body,
        c.  Internal    - Contains 2**ndim children nodes.
    2.  Body            - Element fit into BHTree topology. 
                          It has position in n-dimensions and mass.
"""

import numpy
import json
from operator import attrgetter


class Node:
    """Node is a basic element of a BHTree. It is described by its:
        a. position     - position of center of the cell in n-dim space,
        b. length       - cell extends by length/2 in every direction.
        c. type         - EMPTY, EXTERNAL, INTERNAL
        d. body         - for EXTERNAL nodes only.
                          Bodies are stored as (position = numpy.ndarray, 
                                                mass = float) tuples.
        e. children     - for INTERNAL nodes only. 
                          Children are stored in an array with 2**ndim length.
        f. mass,
        g. center of mass.
    """

    def __init__(self, pos, length):
        assert isinstance(pos, numpy.ndarray), \
            "Position should be a numpy.ndarray."
        assert isinstance(length, (int, float)), \
            "Length should be either a float, or int."

        self.pos = pos
        self.length = length
        self.type = "EMPTY"
        self.body = None
        self.children = None
        self.com = pos
        self.mass = 0
        self.ndim = len(self.pos)

    def fit(self, bodies):
        """Fits a body into the node. 
        Body is either a (pos = numpy.ndarray, mass = float) tuple 
        or a list of them."""

        if not isinstance(bodies, list):
            bodies = [bodies]
        for body in bodies:
            body = Body(body)
            assert body.ndim == self.ndim, \
                "Body and node dimensionality don't match."

            # first check if new body has the same position as the occupant
            if self.type is "EXTERNAL" and numpy.linalg.norm(
                            self.body.pos - body.pos) < 0.00001:
                self.body += body
                return

            # then, check for out of bounds
            bounds_max = self.pos + self.length * 0.5
            bounds_min = self.pos - self.length * 0.5
            if any(body.pos > bounds_max) or any(body.pos < bounds_min):
                print(body.pos, self.pos, self.length)
                raise AssertionError("Body is out of bounds!")

            def child_node_index(_body):
                """Returns an index of a child node 
                from self.children to put body into"""

                # evaluate position of body relative to node's center
                relative_pos = numpy.array(_body.pos > self.pos, dtype=int)
                multiplier = numpy.array([2 ** (self.ndim - 1 - j)
                                          for j in range(self.ndim)])
                index = sum(relative_pos * multiplier)
                return index

            if self.type == "EMPTY":
                self.type = "EXTERNAL"
                self.body = body

            elif self.type == "EXTERNAL":

                # DIVIDE SELF
                # calculate new centers
                offset = self.length * 0.25
                centers = []
                for i in range(2 ** self.ndim):
                    # creates strings like '000', '010', '111' (for ndim=3)
                    s = numpy.binary_repr(i, width=self.ndim)
                    pos = self.pos + [(lambda c: offset if c == '1'
                    else -offset)(c) for c in s]
                    centers.append(pos)

                self.children = [Node(i, self.length * 0.5) for i in centers]

                # find new place for occupant body
                idx = child_node_index(self.body)
                self.children[idx].fit(self.body)
                self.body = None
                self.type = "INTERNAL"

            if self.type == "INTERNAL":
                idx = child_node_index(body)
                try:
                    self.children[idx].fit(body)
                except RecursionError:
                    # just add to existing body
                    self.children[idx].body += body

    def calculate_coms(self):
        """Calculates centers of mass for all nodes."""

        nodes = self._get_all_nodes()
        sorted_nodes = sorted(nodes, key=attrgetter("length"))
        for node in sorted_nodes:
            node._calculate_center_of_mass()

    def _get_all_nodes(self):
        """Used for calculate_coms(). 
        Returns node and all its children's nodes."""

        nodes = []
        if self.type == "INTERNAL":
            for child in self.children:
                nodes += child._get_all_nodes()

        nodes.append(self)
        return nodes

    def _calculate_center_of_mass(self):
        """Used for calculate_coms(). 
        Calculates a center of mass of one node."""

        if self.type == "EMPTY":
            self.com = self.pos
            self.mass = 0
        elif self.type == "EXTERNAL":
            self.com = self.body.pos
            self.mass = self.body.mass
        else:
            sum_pos = numpy.zeros(self.ndim)
            sum_mass = 0
            for child in self.children:
                if child.type == "EMPTY":
                    continue
                if (child.mass == 0) & (child.type == "EXTERNAL"):
                    if child.occupant.mass != 0:
                        print(
                            "Error: Child seems to have wrongly calculated "
                            "mass/center of mass. Recalculating.")
                        child._calculate_center_of_mass()
                sum_pos += child.com * child.mass
                sum_mass += child.mass
            self.com = sum_pos / sum_mass
            self.mass = sum_mass

    def neighbors(self, body, theta=1.2):
        """Returns a list of (position = numpy.ndarray, mass = float) tuples 
        of bodies/nodes affecting a given body.
        Distance is controlled by theta. 
        Lower theta = faster search = less accuracy.
        Body is a (pos = numpy.ndarray, mass = float) tuple.
        """
        body = Body(body)
        assert body.ndim == self.ndim, \
            "Body and node dimensionality don't match."

        if self.type == "EXTERNAL":
            if numpy.array_equal(self.body.pos, body.pos):
                return []
            neighbors = [(self.com, float(self.mass))]
        elif self.type == "INTERNAL":
            dist = numpy.linalg.norm(body.pos - self.com)
            if self.length / dist < theta:
                neighbors = [(self.com, float(self.mass))]
            else:
                neighbors = []
                for child in self.children:
                    neighbors += child.neighbors(body=body, theta=theta)
        else:
            return []

        return neighbors

    def __repr__(self):
        return "<ndbh.Node: %s at %s, length: %d>" % (
            self.type, self.pos, self.length)

    def summary(self, include_empty=False, _final=True):
        """Returns node and all its children in a dictionary form. 
        For debugging / un-black-boxing purposes."""

        return_dict = {'type': self.type, 'pos': str(self.pos.tolist())}

        if self.type != "EMPTY":
            return_dict['center_of_mass'] = str(self.com.tolist())
            return_dict['mass'] = self.mass
            return_dict['length'] = self.length

        if self.type == "INTERNAL":
            children = []
            for child in self.children:
                if child is None:
                    continue
                if (not include_empty) & (child.type == "EMPTY"):
                    continue
                children.append(
                    child.summary(_final=False, include_empty=include_empty))
            return_dict['children'] = children

        if _final:
            return json.dumps(return_dict, indent=4)
        else:
            return return_dict


class Body:
    """Body is an object populating Nodes. It is described by its:
        a. position     - Position in n-dimensional space,
        b. mass.
        """

    def __init__(self, tup):
        if isinstance(tup, Body):
            self.pos = tup.pos
            self.mass = tup.mass
            self.ndim = len(self.pos)

        else:
            assert isinstance(tup, tuple), \
                "Body should be a (pos = numpy.ndarray, mass = float) tuple."
            assert len(tup) == 2, \
                "Body should be a (pos = numpy.ndarray, mass = float) tuple."
            assert isinstance(tup[0], numpy.ndarray), \
                "Position should be numpy.ndarray"
            if isinstance(tup[1], (numpy.int64, numpy.int32,
                                   numpy.int16, numpy.int8,
                                   numpy.float64, numpy.float32,
                                   numpy.float16)):
                tup = (tup[0], numpy.asscalar(tup[1]))
            assert isinstance(tup[1], (int, float)), \
                "Mass should be int or float."

            self.pos = tup[0]
            self.mass = tup[1]
            self.ndim = len(self.pos)

    def __add__(self, other):
        if isinstance(other, self.__class__):
            return Body((self.pos, self.mass + other.mass))
        else:
            raise TypeError("unsupported operand type(s) for +: '{}' and '{}'"
                            ).format(self.__class__, type(other))

    def __repr__(self):
        return "<ndbh.Body: %s, mass: %d>" % (self.pos, self.mass)

    def as_tuple(self):
        tup = (self.pos, self.mass)
        return tup
