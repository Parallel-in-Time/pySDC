This contains the main parts of pySDC.
In ``core``, the basic structure is implemented and abstract classes show how user-level functionality has to be included.
Many implementations of e.g. sweepers, data types or problems can be found in ``implementations``.
These contain the specialized ingredients for a user-defined setup, e.g. an LU-based MLSDC run for the generalized Fisher's equation with Gauss-Radau nodes.
In ``helpers``, we can find helper functions used by the core routines, the implementations or by the user.

Then, in ``playgrounds`` there are various small to medium experiments done with pySDC. They are for example grouped by application (e.g. ``Boris``) or by problem type (e.g. ``ODEs``).
In ``projects`` we gather focussed experiments with pySDC which go beyond simple toy problems or feature tests, while ``tutorial`` contains the files and data for the tutorial codes.