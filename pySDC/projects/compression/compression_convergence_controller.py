from pySDC.core.ConvergenceController import ConvergenceController
import numpy as np

np.bool = np.bool_
import libpressio


class Compression(ConvergenceController):
    def setup(self, controller, params, description, **kwargs):
        default_compressor_args = {
            # configure which compressor to use
            "compressor_id": "sz3",
            # configure the set of metrics to be gathered
            "early_config": {"pressio:metric": "composite", "composite:plugins": ["time", "size", "error_stat"]},
            # configure SZ
            "compressor_config": {
                "pressio:abs": 1e-10,
            },
        }

        defaults = {
            'control_order': 0,
            **super().setup(controller, params, description, **kwargs),
            'compressor_args': {**default_compressor_args, **params.get('compressor_args', {})},
            'min_buffer_length': 12,
        }

        self.compressor = libpressio.PressioCompressor.from_config(defaults['compressor_args'])

        return defaults

    def post_iteration_processing(self, controller, S, **kwargs):
        """
        Replace the solution by the compressed value
        """
        assert len(S.levels) == 1
        lvl = S.levels[0]
        prob = lvl.prob
        nodes = np.append(0, lvl.sweep.coll.nodes)

        encode_buffer = np.zeros(max([len(lvl.u[0]), self.params.min_buffer_length]))
        decode_buffer = np.zeros_like(encode_buffer)

        for i in range(len(lvl.u)):
            encode_buffer[: len(lvl.u[i])] = lvl.u[i][:]
            comp_data = self.compressor.encode(encode_buffer)
            decode_buffer = self.compressor.decode(comp_data, decode_buffer)

            lvl.u[i][:] = decode_buffer[: len(lvl.u[i])]
            lvl.f[i] = prob.eval_f(lvl.u[i], lvl.time + lvl.dt * nodes[i])

            # metrics = self.compressor.get_metrics()
            # print(metrics)
