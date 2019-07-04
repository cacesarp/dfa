import numpy as np
import matplotlib.pyplot as plt

from scipy import stats

class DFA:

    def __init__(self, X, num_points=10):
        self.X = X
        self.Y = None
        self.Fn = None
        self.N = len(self.X)
        self.size_boxs = None
        self.num_points = num_points

    def generate(self):

        mean = np.mean(self.X)
        self.Y = np.cumsum([(x - mean) for x in self.X])
        Y_idxs = np.arange(0, len(self.Y), 1)

        self.Fn = []

        self.size_boxs = [int(l) for l in np.logspace(1, np.log10(self.N + 1), self.num_points)]

        for s in self.size_boxs:

            local_RMS = []

            n_segments = int(len(self.X) / s)

            for segment in range(n_segments):

                index_start = segment * s
                index_stop = (segment + 1) * s

                # linear fit
                slope, intercept, _, _, _ = stats.linregress(Y_idxs[index_start:index_stop], self.Y[index_start:index_stop])
                fit = intercept + slope * Y_idxs[index_start:index_stop]

                # local root mean square
                rms = np.sqrt(np.mean((self.Y[index_start:index_stop] - fit) ** 2))
                local_RMS.append(rms)

            self.Fn.append(np.mean(local_RMS))

        return self.Fn, self.size_boxs

    def view_time_serie(self, scale):
        pass

    def powerlaw_correlation(self, plot_correlation=False):

        slope, intercept, r_value, _, _ = stats.linregress(np.log10(self.size_boxs), np.log10(self.Fn))
        pl_fit = (10 ** intercept) * (self.size_boxs ** slope)

        if plot_correlation:

            plt.xscale('log')
            plt.yscale('log')

            plt.plot(self.size_boxs, self.Fn, 'bo', alpha=0.5)
            plt.plot(self.size_boxs, pl_fit, '--')

            plt.show()

        return slope, r_value**2