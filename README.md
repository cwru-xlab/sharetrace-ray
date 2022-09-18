# ray

Implementations of the ShareTrace algorithm using the [Ray](https://www.ray.io/)
library.

## `main`

This newer implementation applies one-mode projection onto the factor graph to 
obtain a contact-sequence representation of the temporal contact network 
amongst users. Message-passing is performed between subnetwork actors, where 
each actor is responsible for a subset of the users.

### Installing METIS

1. Download [METIS](http://glaros.dtc.umn.edu/gkhome/metis/metis/download). The
   documentation can be
   found [here](http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/manual.pdf).
2. Download [CMake](https://cmake.org/download/) and follow the instructions on
   how to install for command-line use.
3. Run `make config shared=1` in `sharetrace/metis-x.y.z`.

## `factor-graph`

This older implementation still uses the factor graph directly but partitions 
it amongst a set of processors. This implementation performs poorly due to the 
high communication overhead. See below for more details about this 
implementation. The following is a module-level breakdown.

### `app.py`

Contains the "main" program, currently implemented for use by AWS Lambda. This
module provides an implementation of the high-level design described above.

### `model.py`

Contains all the data objects used in the API, such as `RiskScore`,
`Contact`, and `LocationHistory`. Please refer to the module and classes for
further documentation.

### `pda.py`

Contains the `PdaContext` class used to perform asynchronous communication with
contracted PDAs. Note, at this time, it is written with ShareTrace in mind. For
a more general PDA client that is capable of performing CRUD API calls to PDAs,
please see the sharetrace-pda-common, sharetrace-pda-read, and
sharetrace-pda-write directories in the Java API.

### `search.py`

Contains the `ContactSearch` class that searches for `Contact`s among pairs
of `LocationHistory` objects. Please see the module for the extensive
documentation regarding its implementation.

### `graphs.py`

Contains the `FactorGraph` abstract base class and all concrete implementations
using several graph library (and custom) implementations. Currently, the
supported implementations are networkx, igraph, numpy, and Ray. The numpy
implementation simply uses a dictionary to index the vertices, and a numpy array
to store the neighbors of each vertex. The Ray implementation is a "meta"
implementation in that it can use all other implementations, but wraps them as
an actor (see [their documentation](https://ray.io/) for more details).

In addition to the factor graph implementations, the `FactorGraphBuilder` is a
convenience class for specifying several details when constructing factor graphs
and is the recommended way to instantiate the factor graph implementations.

### `stores.py`

Contains the `Queue` abstract base class and concrete implementations.
Currently, the supported implementations are a local queue (wraps
`collections.deque`), async queue (wraps `asyncio.Queue`) and remote queue
(wraps `ray.util.queue.Queue`). A function factory can be used to instantiate
any of these with the specification of a couple of function parameters.

In addition, the `VertexStore` class is used to store vertices and their
optional attributes. The motivation for this class was to utilize the shared
object memory store present in the Ray library, which allows processes to access
the same object. Thus, it is possible to store the graph in the object store by
separating the stateful (the attributes) for the stateless (the structure).
Under the hood, the `VertexStore` is simply a dictionary, but provides two
methods, `get()` and `put()` that provide for several configurations of what is
stored. Please see the class docstring for more details.

### `propagation.py`

Contains the belief propagation implementation. Note that, as of this time,
the `BeliefPropagation` abstract base class is targeted specifically for the
ShareTrace API, as opposed to generic belief propagation setups. There are two
implementations: `LocalBeliefPropagation` and `RemoteBeliefPropagation`. The
former is intended for single-process use and is capable of handling up to 1000
users. Otherwise, the latter should be used because of its utilization of the
Ray multi-processing library. Please see the docstrings for more details.

### `backend.py`

Contains utility globals and functions.
