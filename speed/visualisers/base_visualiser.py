from abc import ABC, abstractmethod

import seaborn as sns
import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np

from speed.metrics.performance_metrics import _concatenate_preds_and_annos
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


class BaseVisualiser(ABC):
    """
    Visualiser for global metrics
    """
    def __init__(self, annotations, predictions):
        self.annotations = annotations
        self.predictions = predictions

        self.colors = sns.color_palette("muted")
        self.current_color = 0
        sns.set_theme(context="paper", style="whitegrid", font_scale=1.5, rc={"lines.linewidth": 2.5})


    def _setup_visualisation(self, nrows=1, ncols=1, sharex=False, sharey=False, num_colors=1):
        """
        Setup visualisation
        """
        fig, ax = plt.subplots(figsize=(10, 8), nrows=nrows, ncols=ncols, sharex=sharex, sharey=sharey)
        colors = self.colors[self.current_color: self.current_color+num_colors]
        self.current_color += num_colors
        if num_colors == 1:
            colors = colors[0]
        return ax, fig, colors

    def _downsample_for_plotting(self, x, y, num_points=5000):
        """
        Uses linear interpolation for downsampling curves for plotting
        """
        f = interpolate.interp1d(x, y)
        new_x = np.linspace(min(x), max(x), num_points)
        return new_x, f(new_x)

    def _get_annos_probs_preds(self):
        if isinstance(self.annotations, list):
            annos, preds, probs = _concatenate_preds_and_annos(self.annotations, self.predictions)
        else:
            probs = self.predictions['probs']
            preds = self.predictions['mask']
            annos = self.annotations
        return annos, probs, preds

    @abstractmethod
    def visualise(self):
        pass


    def _roc_curve(self):
        """
        Visualise ROC curve (Specificity vs Sensitivity)
        """
        annos, probs, _ = self._get_annos_probs_preds()

        fpr, tpr, _ = roc_curve(annos, probs)
        roc_auc = auc(fpr, tpr)

        # Downsample for plotting
        fpr, tpr = self._downsample_for_plotting(fpr, tpr)

        ax, fig, color = self._setup_visualisation()
        sns.lineplot(x=1-fpr, y=tpr, ax=ax, color=color)
        ax.fill_between(1-fpr, tpr, step='post', alpha=0.2, color=color)
        ax.set_xlabel('Specificity')
        ax.set_ylabel('Sensitivity')
        ax.set_title(fr'ROC Curve. $AUC_{{cc}}$ : {roc_auc:.2f}')

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return ax

    def _pr_curve(self):
        """
        Visualise Precision-Recall curve (Sensitivity vs PPV)
        """
        annos, probs, _ = self._get_annos_probs_preds()

        precision, recall, _ = precision_recall_curve(annos, probs)
        average_precision = average_precision_score(annos, probs)

        # Downsample for plotting
        recall, precision = self._downsample_for_plotting(recall, precision)

        ax, fig, color = self._setup_visualisation()
        sns.lineplot(x=recall, y=precision, ax=ax, color=color)
        ax.fill_between(recall, precision, step='post', alpha=0.2, color=color)
        ax.set_xlabel('Recall (Sensitivity)')
        ax.set_ylabel('Precision (PPV)')
        ax.set_title(f'Precision-Recall Curve. AP: {average_precision:0.2f}')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return ax
