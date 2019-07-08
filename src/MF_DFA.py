import numpy as np
import matplotlib.pyplot as plt

from scipy import stats

class MF_DFA:

    def __init__(self, X, num_points=10):
        self.X = X
        self.Y = None
        self.Fn = None
        self.Qs = None
        self.q_Fq = None
        self.N = len(self.X)
        self.scales = None
        self.num_points = num_points

    def run(self):

        mean = np.mean(self.X)
        self.Y = np.cumsum([(x - mean) for x in self.X])
        self.Y_idxs = np.arange(0, len(self.Y), 1)

        F_sq = {}

        # self.Q = np.arange(-5, 6, 1)
        self.Q = list(filter(lambda x: x != 0, np.arange(-5, 6, 1)))

        self.scales = [int(l) for l in np.logspace(1, np.log10(self.N + 1), self.num_points)]

        for s in self.scales:

            local_RMS = []

            n_segments = int(len(self.X) / s)

            for segment in range(n_segments):

                index_start = segment * s
                index_stop = (segment + 1) * s

                # linear fit
                slope, intercept, _, _, _ = stats.linregress(self.Y_idxs[index_start:index_stop],
                                                             self.Y[index_start:index_stop])

                fit = intercept + slope * self.Y_idxs[index_start:index_stop]

                # local root mean square
                rms = np.sqrt(np.mean((self.Y[index_start:index_stop] - fit) ** 2))
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

    def powerlaw_correlation(self, plot_correlation=False):

        q_Hq = {}

        for q in self.q_Fq.keys():

            F = self.q_Fq[q]

            slope, intercept, _, _, _ = stats.linregress(np.log10(self.scales), np.log10(F))
            pl_fit = (10 ** intercept) * (self.scales ** slope)

            q_Hq[q] = slope

            if plot_correlation:

                plt.xscale('log')
                plt.yscale('log')

                plt.plot(self.scales, F, 'o')
                plt.plot(self.scales, pl_fit, '--')

        if plot_correlation:
            plt.show()

        return q_Hq

    def multifractal_spectrum(self, plot_spectrum=False):

        # tau(q) = q*tau(q) - 1
        tau_q = [self.Hq[i] * self.Q[i] - 1 for i in range(len(self.Q))]

        # alpha = tau'(q)
        dy = np.diff(tau_q)
        dx = np.diff(self.Q)
        hq = dy / dx  # alpha

        # Legendre transformation: f(alpha) = q*alpha - tau(q)
        Dq = [self.Q[i] * hq[i] - tau_q[i] for i in range(len(self.Q) - 1)]

        if plot_spectrum:

            plt.xscale('linear')
            plt.yscale('linear')

            plt.subplot(2, 3, 1)
            plt.plot(self.Q, self.Hq, 'ro-')
            plt.xlabel('q-order')
            plt.ylabel('Hq')

            plt.subplot(2, 3, 2)
            plt.plot(self.Q[:-1], hq, 'ro-')
            plt.xlabel('q-order')
            plt.ylabel('hq')

            plt.subplot(2, 3, 3)
            plt.plot(self.Q[:-1], Dq, 'ro-')
            plt.xlabel('q-order')
            plt.ylabel('Dq')

            plt.subplot(2, 2, 3)
            plt.plot(self.Q, tau_q, 'ro-')
            plt.xlabel('q-order')
            plt.ylabel('tq')

            plt.subplot(2, 2, 4)
            plt.plot(hq, Dq, 'ro-')
            plt.xlabel('hq')
            plt.ylabel('Dq')

            plt.show()

        return self.Q, self.Hq, hq, Dq, tau_q