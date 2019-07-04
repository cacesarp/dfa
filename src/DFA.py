import numpy as np

from scipy import stats

class DFA:

    def __init__(self, X, num_points=10):
        self.X = X
        self.Y = None
        self.Fn = None
        self.N = len(self.X)
        self.num_points = num_points

    def generate(self):

        mean = np.mean(self.X)
        self.Y = np.cumsum([(x - mean) for x in self.X])
        Y_idxs = np.arange(0, len(self.Y), 1)

        self.Fn = []

        size_boxs = [int(l) for l in np.logspace(1, np.log10(self.N + 1), self.num_points)]

        for s in size_boxs:

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

        return [self.Fn, size_boxs]

    def view_time_serie(self):
        pass

    def powerlaw_correlation(self, plot_correlation=False):
        pass