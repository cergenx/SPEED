from speed.metrics.performance_metrics import *

class AgreementEvaluator:
    """
    Class for evaluating agreement between annotators and predictions.
    All annotators annoations should be provided
    """
    def __init__(self, annotations, predictions):
        self.annotations = annotations
        self.predictions = predictions
        self._validate_input()

    def _validate_input(self):
        """
        Validate input annotations and predictions.
        """

        if len(self.annotations) != len(self.predictions):
            raise ValueError(f"Number of annotations ({len(self.annotations)}) does not match number of predictions ({len(self.predictions)}).")

        for i, (a, p) in enumerate(zip(self.annotations, self.predictions)):
            if a.shape[1] != p['mask'].shape[0]:
                raise ValueError(f"Annotation shape ({a.shape}) does not match prediction shape ({p['mask'].shape}) for baby {i}.")

    def evaluate(self):
        results = {}

        _mean_per_annotator_ck, _std_per_annotator_ck = cohens_kappa_pairwise(self.annotations, self.predictions)
        results['Pairwise Cohen\'s Kappa mean (std)'] = (_mean_per_annotator_ck, _std_per_annotator_ck)
        _per_annotator_ck_delta = cohens_kappa_pairwise_delta(self.annotations, self.predictions)
        results["Cohen's Kappa Delta"] = _per_annotator_ck_delta
        results['Fleiss\' Kappa'] = fleiss_kappa(self.annotations, self.predictions)
        results['Fleiss\' Kappa Delta'] = fleiss_kappa_delta(self.annotations, self.predictions)
        results['Inferiority Test (p-value)'] = fleiss_kappa_delta_bootstrap(self.annotations, self.predictions, N=1000)

        return results