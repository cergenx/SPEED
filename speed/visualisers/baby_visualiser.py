import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from speed.visualisers.base_visualiser import BaseVisualiser

class BabyVisualiser(BaseVisualiser):
    """
    Visualiser for baby level metrics
    """
    def __init__(self, annotations, predictions, baby_id):
        super().__init__(annotations, predictions)

        self.baby_id = baby_id
        self.annotations = annotations[baby_id]
        self.predictions = predictions[baby_id]

    def visualise(self):
        """
        Visualise global metrics
        """
        self._predictions_for_baby()
        self._seizure_burden_trend()
        if self.annotations.sum() == 0:
            print(f'Warning: No seizures annotated for baby {self.baby_id}, skipping PR and ROC curves.')
        else:
            self._pr_curve()
            self._roc_curve()

        plt.show()

    def _predictions_for_baby(self):
        """
        Get predictions for a baby
        """
        axes, fig, colors = self._setup_visualisation(nrows=2, ncols=1, sharex=True, num_colors=3)
        x = np.arange(len(self.annotations))
        data = {'Prediction': self.predictions['mask'],
                'Probability': self.predictions['probs'],
                'annos': self.annotations,
                'x': x}
        df = pd.DataFrame(data)

        sns.lineplot(x='x', y='annos', data=df, ax=axes[0], color=colors[0], label='Annotations')
        axes[0].set_title(f'Annotations vs Predictions for Baby {self.baby_id}')
        axes[0].set_ylabel('')
        axes[0].legend(title='', loc='upper right')
        axes[0].set_ylim(-.05, 1.05)

        df = df.melt(id_vars='x', value_vars=['Prediction', 'Probability'], var_name='type', value_name='value')
        sns.lineplot(data=df, x='x', y='value', hue='type', ax=axes[1], palette=colors[1:])
        axes[1].set_xlabel('sammples')
        axes[1].set_ylabel('')
        axes[1].legend(title='', loc='upper right')
        axes[1].set_ylim(-.05, 1.05)
        plt.tight_layout()

    def _seizure_burden_trend(self, window_size_mins=20):
        """
        Visualise seizure burden trend
        """
        ax, fig, colors = self._setup_visualisation(num_colors=2)
        pred_seizure_burden = []
        true_seizure_burden = []
        time = []
        idx = 30
        print(f"Sum of annotations: {np.sum(self.annotations)}, len of annotations: {len(self.annotations)}, burden from annotations: {(np.sum(self.annotations)/60)/(len(self.annotations)/3600)}")
        while idx < len(self.predictions['mask']):
            # estimate seizure burden from preceding data (up to 1 hour)
            start = max(0, idx - window_size_mins*60)
            pred_burden = np.mean(self.predictions['mask'][start:idx]) * 60
            true_burden = np.mean(self.annotations[start:idx]) * 60
            pred_seizure_burden.append(pred_burden)
            true_seizure_burden.append(true_burden)
            idx += 30
            time.append(idx)
        time_in_mins = np.array(time) / 60
        pred_seizure_burden = np.array(pred_seizure_burden)
        true_seizure_burden = np.array(true_seizure_burden)

        full_window = time_in_mins > window_size_mins
        sns.lineplot(x=time_in_mins[np.logical_not(full_window)], y=pred_seizure_burden[np.logical_not(full_window)], linestyle='--', ax=ax, color=colors[0])
        sns.lineplot(x=time_in_mins[full_window], y=pred_seizure_burden[full_window], ax=ax, color=colors[0], label='Predicted')
        sns.lineplot(x=time_in_mins[np.logical_not(full_window)], y=true_seizure_burden[np.logical_not(full_window)], linestyle='--', ax=ax, color=colors[1])
        sns.lineplot(x=time_in_mins[full_window], y=true_seizure_burden[full_window], ax=ax, color=colors[1], label='True')
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Seizure Burden (mins/hour)')
        ax.set_title(f'Seizure Burden Trend for Baby {self.baby_id} (est. from {window_size_mins} min window size)')

        return ax
