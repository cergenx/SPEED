from speed.metrics.performance_metrics import *

class ConsensusEvaluator:
    """
    Class for evaluating consensus predictions.
    """
    def __init__(self, annotations, predictions, sample_rate=1):
        self.annotations = annotations
        self.predictions = predictions
        self._validate_input()
        self.sample_freq = sample_rate

    def _validate_input(self):
        """
        Validate input annotations and predictions.
        """

        if len(self.annotations) != len(self.predictions):
            raise ValueError(f"Number of annotations ({len(self.annotations)}) does not match number of predictions ({len(self.predictions)}).")

        for i, (a, p) in enumerate(zip(self.annotations, self.predictions)):
            if a.shape != p['mask'].shape:
                raise ValueError(f"Annotation shape ({a.shape}) does not match prediction shape ({p['mask'].shape}) for baby {i}.")

    def evaluate(self):
        _auc_cc = auc_cc(self.annotations, self.predictions)
        _auc_mean, _auc_std = auc_mean_std(self.annotations, self.predictions)
        _auc_median, _auc_iqr_l, _auc_iqr_u = auc_median_iqr(self.annotations, self.predictions)
        _ap_cc = ap_cc(self.annotations, self.predictions)
        _ap50_cc = ap50_cc(self.annotations, self.predictions)
        _mcc = mcc(self.annotations, self.predictions)
        _pearsons_r = pearsons_r(self.annotations, self.predictions)
        _spearmans_r = spearmans_r(self.annotations, self.predictions)
        _sens = sensitivity(self.annotations, self.predictions)
        _sens_event = sensitivity_event(self.annotations, self.predictions)
        _spec = specificity(self.annotations, self.predictions)
        _ppv = ppv(self.annotations, self.predictions)
        _ppv_event = ppv_event(self.annotations, self.predictions)
        _npv = npv(self.annotations, self.predictions)
        _cohen_kappa = cohens_kappa(self.annotations, self.predictions)
        _sensitivity_baby = sensitivity_baby(self.annotations, self.predictions)
        _specificity_baby = specificity_baby(self.annotations, self.predictions)
        _ppv_baby = ppv_baby(self.annotations, self.predictions)
        _npv_baby = npv_baby(self.annotations, self.predictions)
        _false_detection_per_hour = false_event_detections_per_hour(
            self.annotations, self.predictions, sample_rate=self.sample_freq
        )
        _false_detection_per_hour_non_sz = false_event_detections_per_hour(
            self.annotations, self.predictions, sample_rate=self.sample_freq, non_seizure_cohort_only=True
        )
        _, _, _seizure_burden_corr_sz_only = seizure_burden(self.annotations, self.predictions, seizure_cohort_only=True)
        _, _, _seizure_burden_corr_all = seizure_burden(self.annotations, self.predictions)
        xentropy = cross_entropy(self.annotations, self.predictions)

        results = {
            'AUC_cc': _auc_cc,
            'AUC mean (std)': (_auc_mean, _auc_std),
            'AUC_median (IQR)': (_auc_median, _auc_iqr_l, _auc_iqr_u),
            'AP_cc': _ap_cc,
            'AP50_cc': _ap50_cc,
            'MCC': _mcc,
            'Pearson\'s R': _pearsons_r,
            'Spearman\'s R': _spearmans_r,
            'Sensitivity': _sens,
            'Specificity': _spec,
            'PPV': _ppv,
            'NPV': _npv,
            'Cohen\'s Kappa': _cohen_kappa,
            'Sensitivity (event)': _sens_event,
            'PPV (event)': _ppv_event,
            'Sensitivity (baby)': _sensitivity_baby,
            'Specificity (baby)': _specificity_baby,
            'PPV (baby)': _ppv_baby,
            'NPV (baby)': _npv_baby,
            'False detections / hour (non-szr only)': (_false_detection_per_hour, _false_detection_per_hour_non_sz,),
            'Seizure Burden r (szr only)': (_seizure_burden_corr_all,  _seizure_burden_corr_sz_only),
            'Cross-entropy': xentropy
        }
        return results
