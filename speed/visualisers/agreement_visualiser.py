import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from speed.visualisers.base_visualiser import BaseVisualiser
from speed.metrics.performance_metrics import fleiss_kappa_delta_bootstrap


class AgreementVisualiser(BaseVisualiser):
    """
    Visualiser for global metrics
    """

    def visualise(self):
        """
        Visualise global metrics
        """
        self._kappa_agreement()

        plt.show()


    def _kappa_agreement(self):
        """
        Visualise kappa agreement
        """
        ax, fig, colors = self._setup_visualisation(num_colors=4)
        p, deltas = fleiss_kappa_delta_bootstrap(self.annotations, self.predictions, N=1000, per_annotator=True)
        print(p)
        bins = np.linspace(-.3, .1, 40)
        k_str = r"$\overline{\Delta\kappa}=$"
        sns.histplot(deltas[0, :], bins=bins, alpha=0.3, label=fr'AI, B, C ({k_str}{np.mean(deltas[0]):.3f}, $p=${p[1][0]:.3f})', kde=True, color=colors[0], edgecolor=colors[0], linewidth=1, line_kws={"alpha": 0.7})
        sns.histplot(deltas[1, :], bins=bins, alpha=0.3, label=fr'A, AI, C ({k_str}{np.mean(deltas[1]):.3f}, $p=${p[1][1]:.3f})', kde=True, color=colors[1], edgecolor=colors[1], linewidth=1, line_kws={"alpha": 0.7})
        sns.histplot(deltas[2, :], bins=bins, alpha=0.3, label=fr'A, B, AI ({k_str}{np.mean(deltas[2]):.3f}, $p=${p[1][2]:.3f})', kde=True, color=colors[2],  edgecolor=colors[2], linewidth=1, line_kws={"alpha": 0.7})
        sns.histplot(np.mean(deltas, axis=0), bins=bins, alpha=0.3, label=f'  Mean  ({k_str}{np.mean(deltas):.3f}, $p=${p[0]:.3f})', kde=True, color=colors[3],  edgecolor=colors[3], linewidth=1, line_kws={"alpha": 0.7})
        ax.set_ylabel(None)
        ax.set_yticks([])
        ax.set_xlabel(r"$\Delta \kappa$")
        plt.legend()
