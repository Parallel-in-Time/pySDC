from pySDC.implementations.hooks.default_hook import DefaultHooks


class LearnedQDeltaHook(DefaultHooks):
    """Adds acceptance statistics for learned sweep proposals."""

    def post_sweep(self, step, level_number):
        super().post_sweep(step, level_number)

        L = step.levels[level_number]
        sweep = L.sweep
        if not hasattr(sweep, 'last_used_model'):
            return

        if sweep.last_used_model:
            self.add_to_stats(
                process=step.status.slot,
                process_sweeper=L.sweep.rank,
                time=L.time,
                level=L.level_index,
                iter=step.status.iter,
                sweep=L.status.sweep,
                type='learned_accept',
                value=float(bool(sweep.last_accepted)),
            )
            self.add_to_stats(
                process=step.status.slot,
                process_sweeper=L.sweep.rank,
                time=L.time,
                level=L.level_index,
                iter=step.status.iter,
                sweep=L.status.sweep,
                type='learned_old_residual',
                value=float(sweep.last_old_residual),
            )
            self.add_to_stats(
                process=step.status.slot,
                process_sweeper=L.sweep.rank,
                time=L.time,
                level=L.level_index,
                iter=step.status.iter,
                sweep=L.status.sweep,
                type='learned_trial_residual',
                value=float(sweep.last_trial_residual),
            )
            if hasattr(sweep, 'last_effective_accept_factor'):
                self.add_to_stats(
                    process=step.status.slot,
                    process_sweeper=L.sweep.rank,
                    time=L.time,
                    level=L.level_index,
                    iter=step.status.iter,
                    sweep=L.status.sweep,
                    type='learned_effective_accept_factor',
                    value=float(sweep.last_effective_accept_factor),
                )
            if hasattr(sweep, 'last_confidence_ratio'):
                self.add_to_stats(
                    process=step.status.slot,
                    process_sweeper=L.sweep.rank,
                    time=L.time,
                    level=L.level_index,
                    iter=step.status.iter,
                    sweep=L.status.sweep,
                    type='learned_confidence_ratio',
                    value=float(sweep.last_confidence_ratio),
                )
            if hasattr(sweep, 'last_gate_reason_code'):
                self.add_to_stats(
                    process=step.status.slot,
                    process_sweeper=L.sweep.rank,
                    time=L.time,
                    level=L.level_index,
                    iter=step.status.iter,
                    sweep=L.status.sweep,
                    type='learned_gate_reason_code',
                    value=float(sweep.last_gate_reason_code),
                )
            if hasattr(sweep, 'last_inference_time'):
                self.add_to_stats(
                    process=step.status.slot,
                    process_sweeper=L.sweep.rank,
                    time=L.time,
                    level=L.level_index,
                    iter=step.status.iter,
                    sweep=L.status.sweep,
                    type='learned_inference_time',
                    value=float(sweep.last_inference_time),
                )
            if hasattr(sweep, 'last_trial_eval_time'):
                self.add_to_stats(
                    process=step.status.slot,
                    process_sweeper=L.sweep.rank,
                    time=L.time,
                    level=L.level_index,
                    iter=step.status.iter,
                    sweep=L.status.sweep,
                    type='learned_trial_eval_time',
                    value=float(sweep.last_trial_eval_time),
                )

