ndim_forceatlas
===========

ndim_forceatlas is an implementation of Gephi's ForceAtlas2 algorithm
in Python 3, with option of specifying number of output dimensions.

This package also includes a dimension-agnostic implementation
of Barnes-Hut tree algorithm.

For info on ForceAtlas2, check out this great paper by Mathieu Jacomy,
Tommmaso Venturini, Sebastien Heymann, Mathieu Bastian:

http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0098679

For info on Barnes Hut simulation, check out:

https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation

This implementation of ForceAtlas2 is somewhat inspired by work of Max Shinn, found at:

https://pypi.python.org/pypi/ForceAtlas2/1.0


Foreword
=========

I've written this module because of a personal need of creating a 3D graph.
I'm not a professional programmer, so it may be rough at the edges :)

There's no parallelization implemented. Processing exceptionally large or dense
graphs can be time-consuming.

I've tried including cython compatibility, but couldn't get through python <-> windows
incompatibilities. Maybe next time.


Installation
=========

This module is dependent on Numpy.

Install this module manually: Navigate to project's path and enter:

	python setup.py install


Usage
=========

For ``ndforceatlas``, There's just one function for the end user:

    ndforceatlas.ndforceatlas(g)

Which takes in a adjacency matrix ``g`` and returns a list of position tuples.

Alternatively, for a NetworkX graph as an input:

    ndforceatlas.ndforceatlas_networkx(g)

All parameters of ``ndforceatlas.ndforceatlas()``:

* ``g`` *numpy.ndarray* : adjacency matrix (...or a edges weight matrix) of nodes.

* ``n_dim=2`` : Number of desired output graph dimensions.

* ``n_iter=512`` : Number of iterations.

* ``scaling=2.0`` : Size of the output graph.

* ``gravity=1.0`` : Gravity, used to keep weakly connected nodes from drifting away.
  Increasing ``gravity`` brings them closer to point zero.

* ``use_strong_gravity=False`` : Optional setting.

* ``nodes_barnes_hut_optimization=True`` : Whether Barnes Hut tree algorithm
  should be used when calculating repulsion. Might be counter-productive on smaller graphs.

* ``edges_barnes_hut_optimization=False`` : Whether Barnes Hut tree algorithm
  should be used when calculating attraction. Might be useful in large, dense graphs.
  Using it in small or sparse graphs is discouraged.

* ``barnes_hut_theta=1.2`` : Larger theta = more accuracy = less speed.

* ``edge_weight_influence=1.0`` : Modifies attraction force between nodes.

* ``dissuade_hubs=False`` : Enabling this parameter decreases node clustering in the output graph.

* ``linlog_mode=False`` : With LinLog mode enabled attraction force is calculated logarithmically.
  This way graphs tend to be sparser, with structure of small clusters more apparent.

* ``periodic_autoscale=True`` : Every so often graph will try to scale automatically,
  to converge to final-ish size faster.

* ``verbose=False`` : Enabling this parameter will display progress messages.

Example:

    g = np.array([[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]])
    result = ndforceatlas.forceatlas(g, n_dim=3)
    print(result)

Prints following output:

    [array([ 2.05005147, -2.17161573, -0.47074849]), array([ 0.12278506,  1.03515458,  2.64511372]),
    array([ 0.35797738,  2.14539338, -2.0870159 ]), array([-2.80593661, -1.29705299, -0.5155778 ])]


For ``ndbarneshut`` usage, There are classes/functions:

    ndbarneshut.Node(
        pos,
        length
        )

* ``pos`` : Position as numpy.ndarray, usually numpy.zeros(n_dim)

* ``length`` : Length of an edge of node's hypercube

.

    ndbarneshut.Node.fit(
        bodies
        )

* ``bodies`` : Body or list of bodies, given as a ``(position, mass)`` tuples,
  where ``position`` is a ``numpy.ndarray`` and ``mass`` is an ``int`` or a ``float``.

.

    ndbarneshut.Node.calculate_coms()

* This function calculates all centers of mass of a given node and all its children nodes.
  It has to be called before output can be given.

.

    ndbarneshut.Node.neighbors(
        body,
        theta=1.2
        )

* This function returns a list of ``(position, mass)`` tuples of nodes affecting a given body.

* ``body`` : Body, given as a ``(position, mass)`` tuple,
  where ``position`` is a ``numpy.ndarray`` and ``mass`` is an ``int`` or a ``float``.

* ``theta=1.2`` : Larger theta = more accuracy = less speed.

.

    ndbarneshut.Node.summary(
        include_empty=False,
        _final=True
        )

* Returns node and all its children in a print-friendly form.

* ``include_empty=False`` : By default, all empty nodes are excluded from the summary.

* ``_final=True`` : For internal purposes.


Example:

    tree = ndbarneshut.Node(pos = numpy.array([0, 0, 0]), length = 10)
    for i in range(100000):
        tree.fit([10 * numpy.random.random(3) - 5, numpy.random.random()])
    tree.calculate_coms()
    neighbors = tree.neighbors((numpy.ndarray([-3,4,-1]), 1))

Will return a list of all the nodes (bodies, objects...) affecting a given body.


TO DO
=========

* Include size of nodes and *prevent overlapping* functionality.

* Further optimization? (Cython? Parallelization?)
