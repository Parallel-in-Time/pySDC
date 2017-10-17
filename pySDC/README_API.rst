The Python files contained in ``core``, ``implementations`` and ``helpers`` are fully documented and their API is generated automatically with sphinx-apidoc after each update.
The API (as well as the tutorial pages) can also be generated manually and locally by executing

.. code-block:: none

   > ./docs/update_apidocs.sh
   > nosetests -v --with-id --with-coverage --cover-inclusive --cover-package=core,implementations,helpers,tutorial --cover-html --cover-html-dir=target/doc/build/test_coverage pySDC/tests
   > travis-sphinx build

in the root directory. Then, the main index can be found in ``target/doc/build/index.html``. Anyway, here is the online version:

