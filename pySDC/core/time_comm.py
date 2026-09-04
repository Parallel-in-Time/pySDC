import copy as cp
import logging

from pySDC.core.errors import CommunicationError, ControllerError


class TimeComm(object):
    """
    Abstraction of step ownership and of the communication between steps of one block.

    A controller asks its `TimeComm` which steps it has to work on rather than holding them itself.
    That is the only structural difference between the serial and the parallel PFASST controllers:
    virtually, one process owns every step of the block, under MPI a rank owns `k` of them. Both are
    the same program with a different decomposition parameter.

    **`TimeComm` is a communicator, not a container. It must expose no operation whose natural
    implementation is non-local. If a method's signature would let a caller ask for "all the steps'
    data", it does not belong here.**

    That rule is load bearing rather than stylistic. A gather-shaped method (say
    ``apply_matrix(mat, quantity)``) has an obvious virtual implementation and an MPI implementation
    that costs ``O(L)`` field-sized buffers on every rank, which silently turns a distributed-memory
    code into a replicated one. Every method below is therefore either purely local (`local_steps`),
    nearest-neighbour (`send_forward`, `recv_backward`) or a collective over a scalar (`allreduce`,
    `bcast`, `agree_on_stage`).
    """

    @property
    def size(self):
        """int: number of steps in the current block, counted across all ranks"""
        raise NotImplementedError('TimeComm has to implement size')

    def local_steps(self):
        """
        The steps this rank owns that take part in the current block and have not finished yet. This
        is what the stage machine iterates over; virtually that is up to the whole block, under MPI
        it is at most the `k` steps of this rank.

        Returns:
            list: steps to work on
        """
        raise NotImplementedError('TimeComm has to implement local_steps')

    def send_forward(self, S, level):
        """
        Hand the end point of `S` on `level` to its successor. No-op for the last step of the block.

        Args:
            S (pySDC.core.step.Step): the sending step
            level (int): the level number
        """
        raise NotImplementedError('TimeComm has to implement send_forward')

    def recv_backward(self, S, level):
        """
        Take the end point of the predecessor of `S` on `level` as the new initial value of `S`.
        No-op for the first step of the block and for a step whose predecessor is already done.

        Args:
            S (pySDC.core.step.Step): the receiving step
            level (int): the level number
        """
        raise NotImplementedError('TimeComm has to implement recv_backward')

    def allreduce(self, value, op):
        """
        Reduce one scalar per local step over the whole block and give the result to everyone.

        Args:
            value (list): one entry per local step
            op (callable): reduction, e.g. `all` or `max`

        Returns:
            the reduced value
        """
        raise NotImplementedError('TimeComm has to implement allreduce')

    def bcast(self, value, root):
        """
        Distribute the value held by the step in slot `root` to the whole block.

        Args:
            value (list): one entry per local step
            root (int): slot of the step whose value wins

        Returns:
            the value of the root step
        """
        raise NotImplementedError('TimeComm has to implement bcast')

    def agree_on_stage(self):
        """
        Agree on which stage of the algorithm the block is in. All steps that are not done have to be
        in the same stage, otherwise the block has fallen apart and there is nothing sensible to run.

        Returns:
            str: the common stage
        """
        raise NotImplementedError('TimeComm has to implement agree_on_stage')


class VirtualTimeComm(TimeComm):
    """
    `TimeComm` for the serial controller: this process owns every step, so all communication is a
    copy between two steps in the same memory space and every collective is a Python reduction.

    Attributes:
        steps (list): every step owned here, active or not - `k = L`, so this is the whole block
        block (list): the steps taking part in the block currently being solved
    """

    def __init__(self, steps):
        """
        Args:
            steps (list): the steps to own, in time order
        """
        self.steps = steps
        self.block = list(steps)
        self.logger = logging.getLogger('controller')

    @property
    def size(self):
        return len(self.block)

    def set_block(self, steps):
        """
        Set which of the owned steps take part in the block that is solved next. Called on every
        restart of a block, since steps drop out once their time exceeds `Tend`.

        Args:
            steps (list): the active steps, in time order
        """
        self.block = list(steps)

    def local_steps(self):
        return [S for S in self.block if S.status.stage != 'DONE']

    def send_forward(self, S, level):
        if S.status.last:
            return

        self.logger.debug('Process %2i provides data on level %2i with tag %s' % (S.status.slot, level, S.status.iter))

        # sending here means computing uend ("one-sided communication")
        source = S.levels[level]
        source.sweep.compute_end_point()
        source.tag = cp.deepcopy((level, S.status.iter, S.status.slot))

    def recv_backward(self, S, level):
        if S.status.prev_done or S.status.first:
            return

        self.logger.debug(
            'Process %2i receives from %2i on level %2i with tag %s'
            % (S.status.slot, S.prev.status.slot, level, S.status.iter)
        )

        target, source = S.levels[level], S.prev.levels[level]
        tag = (level, S.status.iter, S.prev.status.slot)
        if source.tag != tag:
            raise CommunicationError('source and target tag are not the same, got %s and %s' % (source.tag, tag))

        # simply do a deepcopy of the values uend to become the new u0 at the target
        target.u[0] = target.prob.dtype_u(source.uend)
        # re-evaluate f on left interval boundary
        target.f[0] = target.prob.eval_f(target.u[0], target.time)

    def allreduce(self, value, op):
        return op(value)

    def bcast(self, value, root):
        return value[root]

    def agree_on_stage(self):
        stages = [S.status.stage for S in self.block if S.status.stage != 'DONE']
        if stages[1:] != stages[:-1]:
            raise ControllerError('not all stages are equal')
        return stages[0]
