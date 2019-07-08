import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from mpl_toolkits import mplot3d
from sklearn.linear_model import LinearRegression

class DFA_2D:

    def __init__(self, X, num_points=10):
        self.X = X
        self.Fn = []
        self.scales = None
        self.num_points = num_points

    def run(self):

        self.Fn = []

        min_scale = min(len(self.X), len(self.X[0]))
        self.scales = [int(l) for l in np.logspace(1, np.log10(min_scale), self.num_points)]

        for s in self.scales:

            Ns = int(len(self.X) / s)
            Ms = int(len(self.X[0]) / s)

            local_RMS = []

            for v in range(Ns):
                for w in range(Ms):

                    M = self.X[v*s : v*s+s, w*s : w*s+s]

                    M = np.reshape(np.cumsum(M), M.shape)

                    ijv = np.array([[v*s+i, w*s+j, M[i][j]] for i in range(s) for j in range(s)])

                    # 2d linear fit

                    x = ijv[:, 0:2]
                    y = ijv[:, -1]

                    model_linear = LinearRegression().fit(x, y)

                    y_pred = model_linear.predict(x)

                    M_fit = np.reshape(y_pred, M.shape)

                    # 2d RMS

                    rms = np.sqrt(np.mean([(M[i][j] - M_fit[i][j])**2 for i in range(s) for j in range(s)]))
                    local_RMS.append(rms)

            self.Fn.append(np.mean(local_RMS))

        return self.Fn, self.scales

    def powerlaw_correlation(self, plot_correlation=False):

        slope, intercept, r_value, _, _ = stats.linregress(np.log10(self.scales), np.log10(self.Fn))
        pl_fit = (10 ** intercept) * (self.scales ** slope)

        if plot_correlation:

            plt.xscale('log')
            plt.yscale('log')

            plt.plot(self.scales, self.Fn, 'bo', alpha=0.5)
            plt.plot(self.scales, pl_fit, '--')

            plt.show()

        return slope, r_value**2

    def view_landscape(self, scale):

        plt.figure()
        ax = plt.axes(projection='3d')

        ijv = np.array([[i, j, self.X[i, j]] for i in range(len(self.X)) for j in range(len(self.X[0]))])

        x = ijv[:, 0:2]
        y = ijv[:, -1]

        a, b = zip(*x)

        ax.plot3D(a, b, y, 'r.')

        Ns = int(len(self.X) / scale)
        Ms = int(len(self.X[0]) / scale)

        for v in range(Ns):
            for w in range(Ms):

                M = self.X[v*scale: v*scale+scale, w*scale: w*scale+scale]

                ijv = np.array([[v*scale+i, w*scale+j, M[i][j]] for i in range(scale) for j in range(scale)])

                x = ijv[:, 0:2]
                y = ijv[:, -1]

                model_linear = LinearRegression().fit(x, y)

                y_pred = model_linear.predict(x)

                M_fit = np.reshape(y_pred, M.shape)

                xx, yy = np.meshgrid([v*scale+k for k in range(scale)], [w*scale+k for k in range(scale)])

                ax.plot_surface(xx, yy, M_fit.transpose(),
                                          linewidth=1, antialiased=False, color='gray', alpha=0.3)

        plt.show()