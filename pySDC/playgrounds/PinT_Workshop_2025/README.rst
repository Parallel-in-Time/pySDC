pySDC Tutorial at 14th PinT Workshop
====================================
**Time**: July 7, 2025

**Place**: ICMS, Edinburgh

Installation
------------
In order to start playing, install `pySDC` and its dependencies, ideally in developer mode.
First, we need to download the repository.
There are multiple ways to do that, but if you plan to work with `pySDC` directly, the best way is to
(1) `fork <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks>`_
the main repository to your Github account and then
(2) `clone <https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository>`_ it from there.
This way you can work on a separate repository while being able to pull updates from the main one and
starting pull requests to merge back your ideas.
You can also clone the main repository, but this will not accept your pushes.
Downloading `pySDC` as a tarball is the easiest, but also the least favorable solution.
Finally, the code can also be obtained using ``pip install``, but then sources are not that easily accessible.

So, please go ahead and clone from your fork on Github:

.. code-block:: bash

    git clone https://github.com/<your_account>/pySDC.git

Next, navigate to the directory that contains this file and setup up a virtual environment, e.g. by using `Micromamba <https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html>`_.
From the root directory of `pySDC`, you can run

.. code-block:: bash

    cd <pySDC-root-dir>/pySDC/playgrounds/PinT_Workshop_2025
     
Now, create the virtual environment with the following command. If you are using ``conda`` instead of ``micromamba``, you can just replace ``micromamba`` with ``conda`` in the commands, or run first run ``conda install -c conda-forge micromamba``.
 
.. code-block:: bash

    micromamba env create -f environment-tutorial.yml
    micromamba activate pySDC_tutorial

     
This may take a while...
Note that this folder and all changes in it will remain even if you leave the virtual environment.
Only installations made with ``micromamba`` or ``pip`` are affected by changing the environment.
Use `branches <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-branches>`_ to isolate development work.
**Please make sure to perform all the following steps inside the virtual environment!**

Finally, you can install `pySDC` via ``pip`` in editable mode using:

.. code-block:: bash

    pip install -e <pySDC-root-dir>

You don't need to do this, but it can make life easier when it comes to setting path variables etc.

Testing
-------

Change to `pySDC`'s root directory and run

.. code-block:: bash

    pytest pySDC/tests -m "base and not slow"

This will check if "all" went well with the installation you just created.
Note that at the time of creating this tutorial we had to remove `mpi4py_fft` from the dependencies.
Therefore, 7 out of the 39 selected test cases may fail!
Anyway, you are now ready to play with `pySDC`.

Jupyter
-------
In order to use our virtual environment within Jupyter, we make a kernel for it with all our nice packages.
We do that with

.. code-block:: bash

    python -m ipykernel install --user --name=pySDC_tutorial

Now, start jupyter and have a look at the notebooks!

.. code-block:: bash

   jupyter notebook
