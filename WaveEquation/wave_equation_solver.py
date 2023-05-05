import numpy as np

class WaveEquationSolver(object):
    def __init__(self, grid_counts: int, grid_interval: float, z0: np.ndarray, c: float, mu: float, dt: float):
        """
        Parameters
        ----------
        grid_counts: int
        grid_interval: float
        z0: numpy.ndarray
            initial potential
        c: float
            phase velocity
        mu: float
            viscosity
        dt: float
            time step
        """

        if grid_counts < 4:
            raise ValueError("Argument 'grid_counts': out of range.")
        if not grid_interval > 0:
            raise ValueError("Argument 'grid_interval': out of range.")
        if not dt < (mu + np.sqrt(mu**2 + 32 * (c / grid_interval)**2)) / (8 * (c / grid_interval)**2):
            raise ValueError("Argument 'dt': too large.")
        if z0.shape != (grid_counts, grid_counts):
            raise ValueError("Argument 'z0': invalid shape.")

        self.__grid_counts = grid_counts
        self.__grid_interval = grid_interval
        self.__z0 = z0
        self.__c = c
        self.__mu = mu
        self.__dt = dt

        h = grid_interval

        self.__c1 = 4 * (1 - 2 * (c * dt / h)**2) / (mu * dt + 2)
        self.__c2 = (mu * dt - 2) / (mu * dt + 2)
        self.__c3 = 2 * (c * dt / h)**2 / (mu * dt + 2)

        self.initialize()

    @property
    def grid_counts(self) -> int:
        """ grid counts
        """
        return self.__grid_counts

    @property
    def grid_interval(self) -> float:
        """ grid interval
        """
        return self.__grid_interval

    @property
    def z0(self) -> np.ndarray:
        """ initial potential
        """
        return self.__z0.copy()

    @property
    def z(self) -> np.ndarray:
        """ potential
        """
        return self.__z.copy()

    @property
    def c(self) -> np.ndarray:
        """ phase velocity
        """
        return self.__c

    @property
    def mu(self) -> np.ndarray:
        """ viscosity
        """
        return self.__mu

    @property
    def dt(self) -> np.ndarray:
        """ time step
        """
        return self.__dt

    @property
    def t(self) -> float:
        """ time
        """
        return self.__t

    def initialize(self) -> None:
        """ Initialize State
        """
        self.__z_prev = self.__z0.copy()
        self.__z = self.__z0.copy()
        self.__t = 0.

    def update(self) -> None:
        """ Update State
        """
        c1, c2, c3 = self.__c1, self.__c2, self.__c3
        z_prev, z = self.__z_prev, self.__z

        z_next = np.zeros_like(z)
        z_next[1:-1, 1:-1] = \
            c1 * z[1:-1, 1:-1] + \
            c2 * z_prev[1:-1, 1:-1] + \
            c3 * (z[:-2, 1:-1] + z[2:, 1:-1] + z[1:-1, :-2] + z[1:-1, 2:])

        self.__z_prev, self.__z = z, z_next
        self.__t += self.__dt

    def batch_update(self, n: int) -> np.ndarray:
        """ Batch Update State
        Parameters
        ----------
        n: int
            step counts
        Returns
        ----------
        zs: numpy.ndarray
            potential progress
        ts: numpy.ndarray
            time
        """

        if n < 1:
            raise ValueError("Argument 'n': must be positive integer.")

        zs = np.empty((n, self.__grid_counts, self.__grid_counts), dtype=float)
        ts = np.empty(n, dtype=float)

        zs[0] = self.__z
        ts[0] = self.__t

        for i in range(1, n):
            self.update()
            zs[i] = self.__z
            ts[i] = self.__t

        return zs, ts

