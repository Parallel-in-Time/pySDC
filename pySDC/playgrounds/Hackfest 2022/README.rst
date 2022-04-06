pySDC Hackfest 2022
===================
**Time**: May 16 + 17, 2022

**Place**: TU Hamburg-Harburg

Installation
------------
In order to start playing, install `pySDC` and it's dependencies, ideally in developer mode.
First start by setting up a virtual environment, e.g. by using `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_:

.. code-block:: text
   > conda create -n pySDC39
   > conda activate pySDC39
   > conda install -c conda-forge --file requirements.txt

When this is done (and it can take a while), download `pySDC` from GitHub.
There are multiple ways to do that, but if you plan to work with pySDC directly, the best way is to
(1) `fork <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks>`_
the main repository to your Github account and then
(2) `clone <https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository>`_ it from there.
This way you can work on a separate repository while being able to pull updates from the main one and
starting pull requests to merge back your ideas.
You can also clone the main repository, but this will not accept your pushes.
Downloading `pySDC` as a tarball is the easiest, but also the least favorable solution.
Finally, the code can also be obtained using ``pip install``, but then sources are not that easily accessible.

So, please go ahead and clone from Github:

.. code-block:: text
    > git clone https://github.com/<your_account>/pySDC.git

Note that this folder and all changes in it will remain even if you leave the virtual environment.
Only installations made with ``conda`` or ``pip`` are affected by changing the environment.
Use `branches <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-branches>`_
to isolate developement work.

Testing
-------

Change to `pySDC`'s root directory and run

.. code-block:: text
    > pytest pySDC/tests

This will check if "all" went well and it will take about 30-60 minutes to complete, depending on your computer.
If all goes well, you are now ready to play with pySDC.

Jupyter
-------
In order to work with the Jupyter notebooks, you first have to install ``jupyter`` with

.. code-block:: text
    > conda install -c conda-forge jupyter

Continue with one of the playgrounds in this directory then.
