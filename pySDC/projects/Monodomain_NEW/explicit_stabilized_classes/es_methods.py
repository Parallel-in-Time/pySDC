import numpy as np
import os


class RKW1:
    def __init__(self, damping, safe_add):
        self.damping = damping
        self.even_s = False
        self.safe_add = safe_add

        paths = os.environ["PYTHONPATH"].split(os.pathsep)
        coeff = None
        for path in paths:
            coeff_file = path + "/pySDC/projects/ExplicitStabilized/explicit_stabilized_classes/coeff_P.csv"
            if os.path.exists(coeff_file):
                coeff = np.loadtxt(coeff_file, delimiter=",", dtype=float, skiprows=1)

        if coeff is None:
            raise Exception("Coefficients file for RKW1 not found.")

        self.s_max = np.max(coeff[:, 0].astype(int))
        self.muP = coeff[:, 1]
        self.nuP = coeff[:, 2]
        self.kappaP = coeff[:, 3]
        self.dP = coeff[:, 4]

    def update_coefficients(self, s):
        self.s = s
        self.mu = self.muP[:s] / self.dP[s - 1]
        self.nu = self.nuP[:s] + self.muP[:s]
        self.kappa = self.kappaP[:s]
        self.nu[0] = s * self.mu[0] / 2.0
        self.kappa[0] = s * self.mu[0]
        self.c = self.dP[:s] / self.dP[s - 1]

    def get_s(self, z):
        if 2.0 * self.dP[-1] <= z:
            raise NotImplementedError("Need too many stages and coefficients are not provided. Try to decrease step size.")
        # if z<1.5:
        #     s = 1
        if z < 2.0:
            s = 1 + self.safe_add
        else:
            s = 1 + np.argmax(2 * self.dP > z) + self.safe_add
            if s > self.s_max:
                raise Exception("adding safe_add will require too many stages and coefficients are not provided")
        return s

    def stability_boundary(self, s):
        if s == 1:
            return 2.0
        else:
            return 2.0 * self.dP[s - 1]


class RKC1:
    def __init__(self, damping, safe_add):
        self.damping = damping
        self.even_s = False
        self.safe_add = safe_add

    def update_coefficients(self, s):
        c = [0] * (s + 1)
        c[s] = 1
        dc = np.polynomial.chebyshev.chebder(c)

        w0 = 1 + self.damping / s**2
        w1 = np.polynomial.chebyshev.chebval(w0, c) / np.polynomial.chebyshev.chebval(w0, dc)

        # evaluate Chebyshev polynomials T_0(w0),...,T_s(w0)
        T = np.zeros(s + 1, dtype=np.float64)
        T[0] = 1.0
        T[1] = w0
        for i in range(2, s + 1):
            T[i] = 2.0 * w0 * T[i - 1] - T[i - 2]

        b = 1.0 / T

        mu = np.zeros(s, dtype=np.float64)
        nu = np.zeros(s, dtype=np.float64)
        kappa = np.zeros(s, dtype=np.float64)
        mu[0] = w1 * b[1]
        mu[1:] = 2.0 * w1 * b[2:] / b[1:-1]
        nu[1:] = 2.0 * w0 * b[2:] / b[1:-1]
        kappa[1:] = -b[2:] / b[0:-2]
        nu[0] = s * mu[0] / 2.0
        kappa[0] = s * mu[0]

        c = np.zeros(s + 1, dtype=np.float64)
        c[0] = 0.0
        c[1] = mu[0]
        for i in range(2, s + 1):
            c[i] = nu[i - 1] * c[i - 1] + kappa[i - 1] * c[i - 2] + mu[i - 1]
        c = c[1:]
        c[-1] = 1.0  # correct eventual roundoff error

        self.s = s
        self.w0 = w0
        self.w1 = w1
        self.mu = mu
        self.nu = nu
        self.kappa = kappa
        self.c = c

    def get_s(self, z):
        # if z<1.5:
        #     s = 1
        if z < 2.0:
            s = 1 + self.safe_add
        else:
            if self.damping <= 0.1:
                beta = 2.0 - 4.0 / 3.0 * self.damping
                s = int(np.ceil(np.sqrt(z / beta)))
            else:
                s = int(np.ceil(np.sqrt(z / 2.0)))
                while z >= self.stability_boundary(s):
                    s += 1
            s += self.safe_add

        return s

    def stability_boundary(self, s):
        # we use the numpy Chebyshev Series module
        # c respresents the series c[0]*T_0(x)+...+c[s]*T_s(x)
        # hence in this case we have only T_s(x)
        c = [0] * (s + 1)
        c[s] = 1
        # get the derivative of this series, i.e. T_s'(x)
        dc = np.polynomial.chebyshev.chebder(c)

        # compute the method coefficients
        w0 = 1 + self.damping / s**2
        w1 = np.polynomial.chebyshev.chebval(w0, c) / np.polynomial.chebyshev.chebval(w0, dc)

        return 2.0 * w0 / w1

    @property
    def get_beta(self):
        if self.damping > 0.5:
            raise Exception("beta in explicit stabilized method is very inaccurate for such damping")
        return 2.0 - 4 / 3.0 * self.damping


class RKU1:
    def __init__(self, damping, safe_add):
        self.damping = damping
        self.even_s = False
        self.safe_add = safe_add

    def update_coefficients(self, s):
        c = [0] * (s + 2)
        c[s + 1] = 1
        c = np.polynomial.chebyshev.chebder(c)
        dc = np.polynomial.chebyshev.chebder(c)

        w0 = 1.0 + 3.0 * self.damping / (s * (s + 1.0) * (s + 2.0))
        w1 = np.polynomial.chebyshev.chebval(w0, c) / np.polynomial.chebyshev.chebval(w0, dc)

        # evaluate Chebyshev polynomials U_0(w0),...,U_s(w0)
        U = np.zeros(s + 1, dtype=np.float64)
        U[0] = 1.0
        U[1] = 2.0 * w0
        for i in range(2, s + 1):
            U[i] = 2.0 * w0 * U[i - 1] - U[i - 2]

        b = 1.0 / U

        mu = np.zeros(s, dtype=np.float64)
        nu = np.zeros(s, dtype=np.float64)
        kappa = np.zeros(s, dtype=np.float64)
        mu[0] = w1 / w0
        mu[1:] = 2.0 * w1 * b[2:] / b[1:-1]
        nu[1:] = 2.0 * w0 * b[2:] / b[1:-1]
        kappa[1:] = -b[2:] / b[0:-2]
        nu[0] = s * mu[0] / 2.0
        kappa[0] = s * mu[0]

        c = np.zeros(s + 1, dtype=np.float64)
        c[0] = 0.0
        c[1] = mu[0]
        for i in range(2, s + 1):
            c[i] = nu[i - 1] * c[i - 1] + kappa[i - 1] * c[i - 2] + mu[i - 1]
        c = c[1:]
        c[-1] = 1.0  # correct eventual roundoff error

        self.s = s
        self.w0 = w0
        self.w1 = w1
        self.mu = mu
        self.nu = nu
        self.kappa = kappa
        self.c = c

    def get_s(self, z):
        # if z<1.5:
        #     s = 1
        if z < 2.0:
            s = 1 + self.safe_add
        else:
            s = int(np.ceil(np.sqrt(1 + 1.5 * z) - 1.0))
            if self.damping > 0.0:
                while z >= self.stability_boundary(s):
                    s += 1
            s += self.safe_add

        return s

    def stability_boundary(self, s):
        c = [0] * (s + 2)
        c[s + 1] = 1
        c = np.polynomial.chebyshev.chebder(c)
        dc = np.polynomial.chebyshev.chebder(c)

        w0 = 1.0 + 3.0 * self.damping / (s * (s + 1.0) * (s + 2.0))
        w1 = np.polynomial.chebyshev.chebval(w0, c) / np.polynomial.chebyshev.chebval(w0, dc)

        return 2.0 * w0 / w1


class HSRKU1:
    def __init__(self, damping, safe_add):
        self.damping = damping
        self.even_s = True
        self.rku = RKU1(damping, 0)
        self.safe_add = safe_add + safe_add % 2

    def update_coefficients(self, s):
        self.rku.update_coefficients(int(s / 2))
        self.rku.nu[0] = 1.0
        self.rku.kappa[0] = 0.0

        self.s = s
        self.mu = np.append(self.rku.mu, self.rku.mu) / 2
        self.nu = np.append(self.rku.nu, self.rku.nu)
        self.kappa = np.append(self.rku.kappa, self.rku.kappa)
        self.c = np.append(self.rku.c, 1.0 + self.rku.c) / 2

    def get_s(self, z):
        s = 2 * self.rku.get_s(z / 2) + self.safe_add
        return s

    def stability_boundary(self, s):
        l = 2 * self.rku.stability_boundary(int(s / 2))
        return l


class mRKC1:
    def __init__(self, outer_class, inner_class, damping, safe_add, scale_separation=True):
        self.damping = damping
        self.safe_add = safe_add

        self.outer_method = eval(outer_class)(damping, safe_add)
        self.inner_method = eval(inner_class)(damping, safe_add)
        self.even_s = self.outer_method.even_s
        self.even_m = self.inner_method.even_s

        self.scale_separation = scale_separation

    def update_coefficients(self, s, m):
        self.s = s
        self.m = m

        self.outer_method.update_coefficients(s)
        self.inner_method.update_coefficients(m)

        self.mu = self.outer_method.mu
        self.nu = self.outer_method.nu
        self.kappa = self.outer_method.kappa
        self.c = self.outer_method.c
        self.delta_c = np.zeros_like(self.c)
        self.delta_c[0] = self.c[0]
        self.delta_c[1:] = self.c[1:] - self.c[:-1]

        self.alpha = self.inner_method.mu
        self.beta = self.inner_method.nu
        self.gamma = self.inner_method.kappa
        self.d = self.inner_method.c
        self.delta_d = np.zeros_like(self.d)
        self.delta_d[0] = self.d[0]
        self.delta_d[1:] = self.d[1:] - self.d[:-1]

    def get_s(self, z):
        return self.outer_method.get_s(z)

    def get_m(self, dt, rho_F, s):
        z = dt * rho_F

        if not self.scale_separation:
            if isinstance(self.inner_method, RKC1):
                betai = self.inner_method.get_beta
                ls = self.outer_method.stability_boundary(s)
                m = int(np.ceil(np.sqrt(1.0 + 6.0 * z / (betai * ls))))
                m = max([m, 2])
                if self.even_m:
                    m += m % 2
                self.eta = dt * 6.0 / ls * m**2 / (m**2 - 1)
            else:
                m = 2
                ls = self.outer_method.stability_boundary(s)
                bound = 2.0 * dt * rho_F / ls
                self.inner_method.update_coefficients(m)
                stab_fun = stability_function(self.inner_method.mu, self.inner_method.nu, self.inner_method.kappa)
                ddR = stab_fun.ddR(0.0)
                lm = self.inner_method.stability_boundary(m)
                while ddR * lm <= bound:
                    m += 1
                    self.inner_method.update_coefficients(m)
                    stab_fun = stability_function(self.inner_method.mu, self.inner_method.nu, self.inner_method.kappa)
                    ddR = stab_fun.ddR(0.0)
                    lm = self.inner_method.stability_boundary(m)
                m = max([m, 2])
                if self.even_m:
                    m += m % 2
                self.eta = 2.0 * dt / ls / ddR
        else:
            ls = self.outer_method.stability_boundary(s)
            self.eta = 2.0 * dt / ls
            m = self.inner_method.get_s(self.eta * rho_F)
            m = max(m, 1)
            if self.even_m:
                m += m % 2

        return m

    def fix_eta(self, dt, s, m):
        ls = self.outer_method.stability_boundary(s)
        if not self.scale_separation:
            if isinstance(self.inner_method, RKC1):
                self.eta = dt * 6.0 / ls * m**2 / (m**2 - 1)
            else:
                self.inner_method.update_coefficients(m)
                stab_fun = stability_function(self.inner_method.mu, self.inner_method.nu, self.inner_method.kappa)
                ddR = stab_fun.ddR(0.0)
                self.eta = 2.0 * dt / ls / ddR
        else:
            self.eta = 2.0 * dt / ls


class stability_function:
    def __init__(self, mu, nu, kappa):
        self.mu = mu
        self.nu = nu
        self.kappa = kappa
        self.s = mu.size

    def R(self, z):
        if self.s == 0:
            return 1.0
        else:
            Rjm1 = 1.0
            Rj = Rjm1 + self.mu[0] * z
            for j in range(2, self.s + 1):
                Rjm2 = Rjm1
                Rjm1 = Rj
                Rj = (self.nu[j - 1] + self.mu[j - 1] * z) * Rjm1 + self.kappa[j - 1] * Rjm2

            return Rj

    def dR(self, z):
        if self.s == 0:
            return 0.0
        else:
            Rjm1 = 1.0
            Rj = Rjm1 + self.mu[0] * z
            dRjm1 = 0.0
            dRj = self.mu[0]
            for j in range(2, self.s + 1):
                Rjm2 = Rjm1
                Rjm1 = Rj
                dRjm2 = dRjm1
                dRjm1 = dRj
                Rj = (self.nu[j - 1] + self.mu[j - 1] * z) * Rjm1 + self.kappa[j - 1] * Rjm2
                dRj = (self.nu[j - 1] + self.mu[j - 1] * z) * dRjm1 + self.kappa[j - 1] * dRjm2 + self.mu[j - 1] * Rjm1

            return dRj

    def ddR(self, z):
        if self.s == 0:
            return 0.0
        else:
            Rjm1 = 1.0
            Rj = Rjm1 + self.mu[0] * z
            dRjm1 = 0.0
            dRj = self.mu[0]
            ddRjm1 = 0.0
            ddRj = 0.0
            for j in range(2, self.s + 1):
                Rjm2 = Rjm1
                Rjm1 = Rj
                dRjm2 = dRjm1
                dRjm1 = dRj
                ddRjm2 = ddRjm1
                ddRjm1 = ddRj
                Rj = (self.nu[j - 1] + self.mu[j - 1] * z) * Rjm1 + self.kappa[j - 1] * Rjm2
                dRj = (self.nu[j - 1] + self.mu[j - 1] * z) * dRjm1 + self.kappa[j - 1] * dRjm2 + self.mu[j - 1] * Rjm1
                ddRj = (self.nu[j - 1] + self.mu[j - 1] * z) * ddRjm1 + self.kappa[j - 1] * ddRjm2 + 2.0 * self.mu[j - 1] * dRjm1

            return ddRj
