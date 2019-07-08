import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.linear_model import LinearRegression

class MF_DFA_2D:

    def __init__(self, X, num_points=10):
        self.X = X
        self.scales = None
        self.Q = None
        self.num_points = num_points

    def run(self):

        F_sq = {}

        self.Q = list(filter(lambda x: x != 0, np.arange(-2, 2, 1)))
        self.Q = [-2,-1,1,2]

        min_scale = min(len(self.X), len(self.X[0]))
        self.scales = [int(l) for l in np.logspace(1, np.log10(min_scale), self.num_points)]

        for s in self.scales:
            local_RMS = []

            Ns = int(len(self.X) / s)
            Ms = int(len(self.X[0]) / s)

            local_RMS = []

            for v in range(Ns):
                for w in range(Ms):

                    M = self.X[v*s: v*s+s, w*s: w*s+s]

                    M = np.reshape(np.cumsum(M), M.shape)

                    ijv = np.array([[v * s + i, w * s + j, M[i][j]] for i in range(s) for j in range(s)])

                    # 2d linear fit

                    x = ijv[:, 0:2]
                    y = ijv[:, -1]

                    model_linear = LinearRegression().fit(x, y)

                    y_pred = model_linear.predict(x)

                    M_fit = np.reshape(y_pred, M.shape)

                    # 2d RMS

                    rms = np.sqrt(np.mean([(M[i][j] - M_fit[i][j]) ** 2 for i in range(s) for j in range(s)]))
                    local_RMS.append(rms)

            local_RMS = np.array(local_RMS)

            for q in self.Q:

                if q != 0:

                    local_qRMS = local_RMS ** q

                    Fq = np.mean(local_qRMS) ** (1 / q)

                    F_sq[(s, q)] = Fq

                else:

                    local_qRMS = np.log(local_RMS)

                    Fq = np.exp( 0.5 * np.mean(local_qRMS) )

                    F_sq[(s, q)] = Fq

        self.q_Fq = {}
        self.Hq = []

        for q in self.Q:
            print(q)
            F = []
            for s in self.scales:
                F.append(F_sq[s, q])

            self.q_Fq[q] = F

            slope, intercept, _, _, _ = stats.linregress(np.log10(self.scales), np.log10(F))

            self.Hq.append(slope)

        return self.scales, self.q_Fq

    def powerlaw_correlation(self):
        pass