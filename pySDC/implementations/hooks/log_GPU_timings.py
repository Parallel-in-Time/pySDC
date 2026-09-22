import cupy as cp

from pySDC.core.timings import Timings


class GPUTimings(Timings):
    """
    Hook for recording GPU timings of important operations during a pySDC run.
    """

    prefix = 'GPU_'

    def _compute_time_elapsed(self, event_after, event_before):
        event_after.synchronize()
        return cp.cuda.get_elapsed_time(event_before, event_after) / 1e3

    def _get_event(self):
        event = cp.cuda.Event()
        event.record()
        return event
