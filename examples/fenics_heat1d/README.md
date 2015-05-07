FEniCS on OSX
=============

To get FEniCS running on an OSX installation, download the binary package from the FEniCS website annd install it.
Then make sure that Apple's system python is the first in your PATH, e.g. by using `export PATH=/usr/bin:$PATH`.
In particular, this is important when Anaconda is normally used.

A working FEniCS environment can then be started with

    /Applications/FEniCS.app/Contents/Resources/bin/fenics-terminal
    
or, combining this with the modification of the PATH variable,

    PATH=/usr/bin:$PATH /Applications/FEniCS.app/Contents/Resources/bin/fenics-terminal
    
This should get you a new console with everything set up to run FEniCS examples. 
Test this by checking out one for the examples from /Applications/FEniCS.app/Contents/Resources/share/dolfin/demo/.

However, pySDC will most likely fail to run. This is due to missing our outdated packages and can be fixed by running

    sudo easy_install future
    sudo easy_install numpy==1.9

Note that it is crucial to run the correct version of easy_install, i.e. the one from Apple's system python 
located in /usr/bin. This is also the reason why using pip does not seem to work.

Finally, test pySDC by running e.g. the heat1d example with
    
    export PYTHONPATH=$PYTHONPATH:../..
    python playground.py
    
Note that PYTHONPATH already contains FEniCS-related directories which need to be preserved.
    
