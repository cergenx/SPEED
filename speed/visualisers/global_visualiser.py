import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from speed.visualisers.base_visualiser import BaseVisualiser
from speed.metrics.performance_metrics import _event_prediction_stats, seizure_burden


class GlobalVisualiser(BaseVisualiser):
    """
    Visualiser for global metrics
    """

    def visualise(self):
        """
        Visualise global metrics
        """
        self._event_predictions()
        self._roc_curve()
        self._pr_curve()
        self._seizure_burden()

        plt.show()

    def _seizure_burden(self):
        """
        Visualise seizure burden
        """
        self.current_color += 1
        true_seizure_burden, pred_seizure_burden, r = seizure_burden(self.annotations, self.predictions, sample_rate=self.sample_freq)
        df = pd.DataFrame({'True Seizure Burden': true_seizure_burden, 'Predicted Seizure Burden': pred_seizure_burden})
        grid = sns.lmplot(data=df, x='True Seizure Burden', y='Predicted Seizure Burden', height=8, aspect=1.25,
                          scatter_kws={'color': self.colors[self.current_color]},
                          line_kws={'color': self.colors[self.current_color]})

        ax = grid.axes.flat[0]
        ax.text(0.05, 0.95, f'R={r:.2f}', ha='left', va='top', transform=ax.transAxes)
        ax.set_xlim(-1, 61)
        ax.set_ylim(-1, 61)

        return ax

    def _event_predictions(self):
        """
        Get event predictions
        """

        annos, _, preds = self._get_annos_probs_preds()

        predicted_events_detection, predicted_event_duration, true_events_detection, true_event_duration = _event_prediction_stats(annos, preds, sample_rate=self.sample_freq)

        # exponentially growing bin sizes
        num_bins = 15
        max_duration = max(predicted_event_duration.max(), true_event_duration.max())
        exp_values = np.linspace(0, np.log(max_duration), num_bins)
        bin_edges = np.exp(exp_values)

        def generate_custom_ticks(sparse_factor):
            ticks = [0]
            spacing = 30
            max_ticks_at_spacing = 2
            while ticks[-1] < max_duration:
                next_tick = ticks[-1] + spacing
                if len(ticks) % max_ticks_at_spacing == 0:
                    spacing *= sparse_factor  # Increase the spacing exponentially
                ticks.append(next_tick)
            ticks[0] = 10
            ticks = np.array([1] + ticks[:-1])
            return ticks

        ticks = generate_custom_ticks(3)

        axes, fig, _ = self._setup_visualisation(ncols=1, nrows=2)
        colors = [self.colors[0], self.colors[3], self.colors[2]]
        def _plot_events_by_duration(detection, duration, ax, pred=False):
            df = pd.DataFrame({'Detection': detection,'Duration': duration})
            df['DurationBin'] = pd.cut(df['Duration'], bins=bin_edges, labels=False, include_lowest=True)
            df_grouped = df.groupby('DurationBin')['Detection'].value_counts(normalize=False).unstack().fillna(0)

            # Reindex to include all bins
            complete_index = pd.Index(range(num_bins-1), name='DurationBin')
            df_grouped_reindexed = df_grouped.reindex(complete_index, fill_value=0)

            # Extract counts for plotting
            y_detected = df_grouped_reindexed[1].values
            y_undetected = df_grouped_reindexed[0].values
            y_total = y_detected + y_undetected

            x = (bin_edges[1:] + bin_edges[:-1]) / 2

            data = {'Duration': x, 'Detected': y_detected, 'Undetected': y_undetected}
            df = pd.DataFrame(data)
            ax.set_xscale('log')
            acc = y_detected *100 / y_total
            ax2 = ax.twinx()
            ax2.set_ylim(0, 101)
            sns.lineplot(x=x, y=acc, ax=ax2, color=colors[0], linestyle=':', label='Detection Rate', linewidth=3)

            df = df.melt(id_vars='Duration', value_vars=['Detected', 'Undetected'], var_name='Detection', value_name='Count')
            sns.lineplot(x='Duration', y='Count', data=df, hue='Detection', ax=ax, err_style='bars', linewidth=2.5,
                          linestyle='-',marker='o', markersize=12, palette=colors[1:], legend='brief')

            sns.move_legend(ax, loc='lower left')
            ax.set_ylabel('Event Count')
            ax.set_xlabel('Seizure event duration in seconds (log scale)')
            ax.set_title(f'{"Predicted " if pred else ""}Seizure Event {"Detection " if not pred else ""}by Duration')

            ax2.set_ylabel('Detection Rate (%)')


            ax.set_xticks(ticks, [f"{tick}" for tick in ticks])

            ax.tick_params(axis='both', which='major', bottom=True, left=True)
            ax.set_xlim(1, max_duration)
            ax.set_ylim(0, max(y_total) * 1.1)

            handles1, labels1 = ax.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            handles = handles1 + handles2
            labels = labels1 + labels2
            ax.legend(handles, labels, loc='upper left')
            ax2.legend().remove()
            ax.set_xticks(ticks, [f"{tick}" for tick in ticks])
        _plot_events_by_duration(predicted_events_detection, predicted_event_duration, axes[0], pred=True)
        _plot_events_by_duration(true_events_detection, true_event_duration, axes[1])
        plt.tight_layout()
