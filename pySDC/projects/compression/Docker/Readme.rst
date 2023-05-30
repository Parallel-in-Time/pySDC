Instructions for using libpressio in the Docker container
---------------------------------------------------------

As docker desktop is no longer available for commercial use for free, you may need to install an alternative, such as `colima <https://github.com/abiosoft/colima>`_ first.

If you haven't done this already, build the container using

.. code-block:: bash
    
    cd <local_path_to_pySDC>/pySDC/projects/compression/Docker
    docker build -t libpressio .
 
This creates an image with the name 'libpressio'.
Please pay attention to the platform you are using and you intend to run on. If you use this command on an ARM machine and try to use the image in a GitHub action, it will not run because it requires AMD architecture. You can build a platform specific version for GitHub using

.. code-block:: bash

    docker buildx build --platform linux/amd64 -t libpressio:amd64 .


If you are on an ARM machine like me, replace `amd64` by `arm64` to build an image for your local machine. Remember to replace the tag with something useful, such as  ``-t libpressio:arm64``.
 
Start the image using

.. code-block:: bash

    docker run -v <local_absolute_path_to_pySDC_installation>:/pySDC -ti --rm libpressio


You may have to change the tag to the platform specific version.
The `-v` does a `"bind_mount" <https://docs.docker.com/storage/bind-mounts/>`_ to pySDC on your local machine.
We want that because it let's us access the same version of pySDC that we have locally inside the container, in particular with all modifications that we make while the container is running.
The ``-ti`` flag opens the image in an interactive manner, which allows us to run things inside the container and the ``--rm`` flag removes the image once we are done with it.

We have specified an entry point in the Docker file which will install the local version of pySDC using ``pip``.
If you run into trouble, you may consult the file ``Docker/docker-entrypoint.sh`` in the compression project folder for what is required to install pySDC.
Keep in mind that spack wants its own python, which means we are not working with Conda here. Just use ``pip`` to install more dependencies. You can also add ``pip`` commands to the entry point file in order to make persistent changes to the container or you can create a new Dockerfile based on the current one and replace the entry point by whatever you want if you're doing something non-generic.

Have fun!

TODOs
_____
 - Streamline the multiplatform business. See, for instance `here <https://docs.docker.com/build/building/multi-platform/>`_.
