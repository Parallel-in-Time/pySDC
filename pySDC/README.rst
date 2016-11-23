This contains the main parts of pySDC.
In ``core``, the basic structure is implemented and abstract classes show how user-level functionality has to be included.
Many implementations of e.g. sweepers, data types or problems can be found in ``implementations``.
These contain the specialized ingredients for a user-defined setup, e.g. an LU-based MLSDC run for the generalized Fisher's equation with Gauss-Radau nodes.
In ``helpers``, we can find helper functions used by the core routines, the implementations or by the user.

The Python files contained in these three packages are fully documented and their API is generated automatically with sphinx-apidoc after each update.
The API (as well as the tutorial pages) can also be generated manually and locally by executing

.. code-block:: none

   > ./docs/update_apidocs.sh
   > nosetests tests -v --with-id --with-coverage --cover-inclusive --cover-package=pySDC,tutorial --cover-html --cover-html-dir=target/doc/build/test_coverage
   > travis-sphinx build

in the root directory. Then, the main index can be found in ``target/doc/build/index.html``. Anyway, here is the online version:

