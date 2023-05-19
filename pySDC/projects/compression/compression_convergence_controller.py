from pySDC.core.ConvergenceController import ConvergenceController
from pySDC.projects.compression.compressed_mesh import compressed_mesh
import numpy as np
import libpressio


class Compression(ConvergenceController):
    def setup(self, controller, params, description, **kwargs):
        default_compressor_args = {
            # configure which compressor to use
            "compressor_id": "sz3",
            # configure the set of metrics to be gathered
            "early_config": {
                "pressio:metric": "composite",
                "composite:plugins": ["time", "size", "error_stat"],
            },
            # configure SZ
            "compressor_config": {
                "pressio:abs": 1e-10,
            },
        }

        defaults = {
            "control_order": 0,
            **super().setup(controller, params, description, **kwargs),
            "compressor_args": {
                **default_compressor_args,
                **params.get("compressor_args", {}),
            },
            "min_buffer_length": 12,
        }

        self.compressor = libpressio.PressioCompressor.from_config(
            defaults["compressor_args"]
        )

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


class Compression_Conv_Controller(ConvergenceController):
    def setup(self, controller, params, description, **kwargs):
        defaults = {
            "control_order": 0,
            "errBound": 1,
            **super().setup(controller, params, description, **kwargs),
        }

        # The bottom line gets access to manager but makes a new mesh
        # x = compressed_mesh(init=((30,), None, np.float64))
        # self.manager = x.manager
        self.manager = compressed_mesh(init=((30,), None, np.float64)).manager
        self.manager.errBound = defaults["errBound"]
        return defaults

    def dependencies(self, controller, description, **kwargs):
        """
        Load estimator of embedded error.

        Args:
            controller (pySDC.Controller): The controller
            description (dict): The description object used to instantiate the controller

        Returns:
            None
        """

        from pySDC.implementations.convergence_controller_classes.estimate_contraction_factor import (
            EstimateContractionFactor,
        )

        controller.add_convergence_controller(
            EstimateContractionFactor,
            description=description,
        )

    def post_iteration_processing(self, controller, S, **kwargs):
        self.log(S.levels[0].status.contraction_factor, S, level=10)
        self.log(S.levels[0].status.error_embedded_estimate, S)

    # def post_step_processing(self, controller, S, **kwargs):
    #     print(self.manager)
