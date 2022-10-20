import struct
from datetime import datetime

import numpy as np

from pySDC.helpers.pysdc_helper import FrozenClass
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit


class _fault_stats(FrozenClass):
    def __init__(self):
        self.nfaults_called = 0
        self.nfaults_injected_u = 0
        self.nfaults_injected_f = 0
        self.nfaults_detected = 0
        self.ncorrection_attempts = 0
        self.nfaults_missed = 0
        self.nfalse_positives = 0
        self.nfalse_positives_in_correction = 0
        self.nclean_steps = 0

        self._freeze()


class implicit_sweeper_faults(generic_implicit):
    """
    LU sweeper using LU decomposition of the Q matrix for the base integrator, special type of generic implicit sweeper

    """

    def __init__(self, params):
        """
        Initialization routine for the custom sweeper

        Args:
            params: parameters for the sweeper
        """

        if 'allow_fault_correction' not in params:
            params['allow_fault_correction'] = False

        if 'detector_threshold' not in params:
            params['detector_threshold'] = 1.0

        if 'dump_injections_filehandle' not in params:
            params['dump_injections_filehandle'] = None

        # call parent's initialization routine
        super(implicit_sweeper_faults, self).__init__(params)

        self.fault_stats = _fault_stats()

        self.fault_injected = False
        self.fault_detected = False
        self.in_correction = False
        self.fault_iteration = False

    def reset_fault_stats(self):
        """
        Helper method to reset all fault related stats and flags. Will be called after the run in post-processing.
        """

        self.fault_stats = _fault_stats()
        self.fault_injected = False
        self.fault_detected = False
        self.in_correction = False
        self.fault_iteration = False

    @staticmethod
    def bitsToFloat(b):
        """
        Static helper method to get a number from bit into float representation

        Args:
            b: bit representation of a number

        Returns:
            float representation of b
        """
        s = struct.pack('>q', b)
        return struct.unpack('>d', s)[0]

    @staticmethod
    def floatToBits(f):
        """
        Static helper method to get a number from float into bit representation

        Args:
            f: float representation of a number

        Returns:
            bit representation of f
        """
        s = struct.pack('>d', f)
        return struct.unpack('>q', s)[0]

    def do_bitflip(self, a, pos):
        """
        Method to do a bit flip

        Args:
            a: float representation of a number
            pos (int between 0 and 63): position of bit flip

        Returns:
            float representation of a number after bit flip at pos
        """
        # flip of mantissa (fraction) bit (pos between 0 and 51) or of exponent bit (pos between 52 and 62)
        if pos < 63:
            b = self.floatToBits(a)
            # mask: bit representation with 1 at pos and 0 elsewhere
            mask = 1 << pos
            # ^: bitwise xor-operator --> bit flip at pos
            c = b ^ mask
            return self.bitsToFloat(c)
        # "flip" of sign bit (pos = 63)
        elif pos == 63:
            return -a

    def inject_fault(self, type=None, target=None):
        """
        Main method to inject a fault

        Args:
            type (str): string describing whether u of f should be affected
            target: data to be modified
        """

        pos = 0
        bitflip_entry = 0

        # do bitflip in u
        if type == 'u':

            # do something to target = u here!
            # do a bitflip at random vector entry of u at random position in bit representation
            ulen = len(target)
            bitflip_entry = np.random.randint(ulen)
            pos = np.random.randint(64)
            tmp = target[bitflip_entry]
            target[bitflip_entry] = self.do_bitflip(target[bitflip_entry], pos)
            # print('     fault in u injected')

            self.fault_stats.nfaults_injected_u += 1

        # do bitflip in f
        elif type == 'f':

            # do something to target = f here!
            # do a bitflip at random vector entry of f at random position in bit representation
            flen = len(target)
            bitflip_entry = np.random.randint(flen)
            pos = np.random.randint(64)
            tmp = target[bitflip_entry]
            target[bitflip_entry] = self.do_bitflip(target[bitflip_entry], pos)
            # print('     fault in f injected')

            self.fault_stats.nfaults_injected_f += 1

        else:

            tmp = None
            print('ERROR: wrong fault type specified, got %s' % type)
            exit()

        self.fault_injected = True

        if self.params.dump_injections_filehandle is not None:
            out = str(datetime.now())
            out += ' --- '
            out += type + ' ' + str(bitflip_entry) + ' ' + str(pos)
            out += ' --- '
            out += str(tmp) + ' ' + str(target[bitflip_entry]) + ' ' + str(np.abs(tmp - target[bitflip_entry]))
            out += '\n'
            self.params.dump_injections_filehandle.write(out)

    def detect_fault(self, current_node=None, rhs=None):
        """
        Main method to detect a fault

        Args:
            current_node (int): current node we are working with at the moment
            rhs: right-hand side vector for usage in detector
        """

        # get current level for further use
        L = self.level

        # calculate solver residual
        res = L.u[current_node] - L.dt * self.QI[current_node, current_node] * L.f[current_node] - rhs
        res_norm = np.linalg.norm(res, np.inf)
        if np.isnan(res_norm) or res_norm > self.params.detector_threshold:
            # print('     FAULT DETECTED!')
            self.fault_detected = True
        else:
            self.fault_detected = False

        # update statistics
        # fault injected and fault detected -> yeah!
        if self.fault_injected and self.fault_detected:
            self.fault_stats.nfaults_detected += 1
        # no fault injected but fault detected -> meh!
        elif not self.fault_injected and self.fault_detected:
            self.fault_stats.nfalse_positives += 1
            # in correction mode and fault detected -> meeeh!
            if self.in_correction:
                self.fault_stats.nfalse_positives_in_correction += 1
        # fault injected but no fault detected -> meh!
        elif self.fault_injected and not self.fault_detected:
            self.fault_stats.nfaults_missed += 1
        # no fault injected and no fault detected -> yeah!
        else:
            self.fault_stats.nclean_steps += 1

    def correct_fault(self):
        """
        Main method to correct a fault or issue a restart
        """

        # do correction magic or issue restart here... could be empty!

        # we need to make sure that not another fault is injected here.. could also temporarily lower the probability
        self.in_correction = True
        # print('     doing correction...')

        self.fault_stats.ncorrection_attempts += 1
        self.fault_detected = False

    def update_nodes(self):
        """
        Update the u- and f-values at the collocation nodes -> corresponds to a single sweep over all nodes

        Returns:
            None
        """

        # get current level and problem description
        L = self.level
        P = L.prob

        # only if the level has been touched before
        assert L.status.unlocked

        # get number of collocation nodes for easier access
        M = self.coll.num_nodes

        # gather all terms which are known already (e.g. from the previous iteration)
        # this corresponds to u0 + QF(u^k) - QdF(u^k) + tau

        # get QF(u^k)
        integral = self.integrate()
        for m in range(M):

            # get -QdF(u^k)_m
            for j in range(M + 1):
                integral[m] -= L.dt * self.QI[m + 1, j] * L.f[j]

            # add initial value
            integral[m] += L.u[0]
            # add tau if associated
            if L.tau[m] is not None:
                integral[m] += L.tau[m]

        fault_node = np.random.randint(M)

        # do the sweep
        m = 0
        while m < M:

            # see if there will be a fault
            self.fault_injected = False
            fault_at_u = False
            fault_at_f = False
            if not self.in_correction and m == fault_node and self.fault_iteration:
                if np.random.randint(2) == 0:
                    fault_at_u = True
                else:
                    fault_at_f = True

            # build rhs, consisting of the known values from above and new values from previous nodes (at k+1)
            # this is what needs to be protected separately!
            rhs = P.dtype_u(integral[m])
            for j in range(m + 1):
                rhs += L.dt * self.QI[m + 1, j] * L.f[j]

            if fault_at_u:

                # implicit solve with prefactor stemming from the diagonal of Qd
                L.u[m + 1] = P.solve_system(
                    rhs, L.dt * self.QI[m + 1, m + 1], L.u[m + 1], L.time + L.dt * self.coll.nodes[m]
                )

                # inject fault at some u value
                self.inject_fault(type='u', target=L.u[m + 1])

                # update function values
                L.f[m + 1] = P.eval_f(L.u[m + 1], L.time + L.dt * self.coll.nodes[m])

            elif fault_at_f:

                # implicit solve with prefactor stemming from the diagonal of Qd
                L.u[m + 1] = P.solve_system(
                    rhs, L.dt * self.QI[m + 1, m + 1], L.u[m + 1], L.time + L.dt * self.coll.nodes[m]
                )

                # update function values
                L.f[m + 1] = P.eval_f(L.u[m + 1], L.time + L.dt * self.coll.nodes[m])

                # inject fault at some f value
                self.inject_fault(type='f', target=L.f[m + 1])

            else:

                # implicit solve with prefactor stemming from the diagonal of Qd
                L.u[m + 1] = P.solve_system(
                    rhs, L.dt * self.QI[m + 1, m + 1], L.u[m + 1], L.time + L.dt * self.coll.nodes[m]
                )

                # update function values
                L.f[m + 1] = P.eval_f(L.u[m + 1], L.time + L.dt * self.coll.nodes[m])

            # see if our detector finds something
            self.detect_fault(current_node=m + 1, rhs=rhs)

            # if we are allowed to try correction, do so, otherwise proceed with sweep
            if not self.in_correction and self.fault_detected and self.params.allow_fault_correction:
                self.correct_fault()
            else:
                self.in_correction = False
                m += 1

        # indicate presence of new values at this level
        L.status.updated = True

        return None
