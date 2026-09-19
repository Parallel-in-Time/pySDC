from pySDC.implementations.hooks.AllenCahn_monitor import AllenCahnMonitor


class monitor(AllenCahnMonitor):
    """The MPIFFT problems put the wells at 0 and 1, so the field integrates straight to the volume."""

    calibrate = True
