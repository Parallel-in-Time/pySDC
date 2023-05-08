Parallel-in-time simulation for multi-modal energy systems
==========================================================

In this project, we want to model energy systems as systems of ordinary differential equations to simulate them in parallel in time. Here, the focus lies on a simulation faster than real-time. Our first energy system that we consider is the DC microgrid, which consists of a Pi-model transmission line, buck (step-down) converters and a battery drain model. In the last two components there occur discrete events due to switching processes and we are interested in how SDC and PFASST can handle with it.

Switching processes
-------------------
In our microgrid, buck converters emulate the power consumption behavior of a household. They convert a large input voltage into a smaller target output voltage, which is done by multiple switchings. In each step, the actual output value will be compared with a target output voltage. Based in the error, the duty cycle is recalculated to control the switching process. 
The battery drain model as the another challenging component in microgrid, energy will provided by a capacitor for the first time. The capacitor discharge after time and when its voltage drops below a reference value, the battery switches to another voltage source. Here, only one switch occurs and it is perfectly to investigate it in SDC and PFASST.
Looking at the switching processes, an interesting question arises: How can SDC (and PFASST) deal with switching? Recent investigations yield that it does not matter which time step is used. Switching on a time step leads to no complications. Switching inside a time subinterval results in a lower residual after the switch. In power system engineering, good accuracy is crucial to simulate energy grids. Therefore, a detection and handling must be found.  

The pi-line test case
---------------------
The plot below shows the simulation of the pi-line model. There can be seen the two voltages along the capacitor and the current along the coil. Both the voltages and the current settle down over the time. As a background information: the pi-line model serves as transmission line, which transports the energy. The behavior which can be seen in the plot is what we would expect. 

.. image:: ../../../docs/build/html/data/piline_model_solution.png
    :width: 35%
    :align: center


The buck converter test case
----------------------------
In the test case of the buck converter there are multiple switches in the considered time domain. In the so-called open-loop control, a controller monitors the actual output voltage. It compares the output with a target output voltage. Regularly, after a fixed number of time steps, the duty cycle to control the switching is recalculated based on the error. The simulation illustrates the switching behavior in the simulation: The voltage values settle down between the reference. 

.. image:: ../../../docs/build/html/data/buck_model_solution.png
    :width: 35%
    :align: center

The battery drain model test case
---------------------------------
This model is a simple example for system internal switching, which means switching depends on system dynamics. When the capacitor voltage drops below the reference value, the circuit switches to the voltage source. An illustration of the process is clarified in the plot below: The capacitor voltage drops down, and after switching the voltage value along the capacitor no longer changes. 

.. image:: ../../../docs/build/html/data/battery_model_solution.png
    :width: 35%
    :align: center
