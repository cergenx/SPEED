import numpy as np

from sklearn.metrics import roc_auc_score, matthews_corrcoef, recall_score, precision_score, cohen_kappa_score, average_precision_score, precision_recall_curve, log_loss
from scipy.stats import pearsonr, spearmanr, norm
from statsmodels.stats.inter_rater import fleiss_kappa as _fleiss_kappa
from tqdm import tqdm


def _format(metric):
    """
    Format a metric as a float with 3 decimal places.
    :param metric: Metric to format.
    :return: Formatted metric.
    """
    rounded = round(metric, 3)
    if isinstance(rounded, (np.float64, np.int64)):
        return rounded.item()
    else:
        return rounded

def _verify_mulitple_annotators(annotations):
    """
    Verify that there are multiple annotators for each baby.
    :param annotations: Annotations for each baby.
    """
    if len(annotations[0].shape) != 2:
        raise ValueError("Annotations should be a list of 2D arrays, where each array is the annotations for a single baby from each annotator.")

def _concatenate_preds_and_annos(annotations, predictions):
    """
    Concatenate all annotations and predictions.
    :param annotations: Annotations for each baby.
    :param predictions: Predictions for each baby.
    :return: Concatenated annotations and predictions.
    """
    all_annotations = np.concatenate(annotations)
    all_preds = np.concatenate([pred['mask'] for pred in predictions])
    all_probs = np.concatenate([pred['probs'] for pred in predictions])
    return all_annotations, all_preds, all_probs

def auc_cc(annotations, predictions):
    """
    Calculate the AUC across all babies, by concatenating all annotations and predictions.
    :param annotations: Annotations for each baby.
    :param predictions: Predictions for each baby.
    :return: AUC
    """
    all_annotations, _, all_probs = _concatenate_preds_and_annos(annotations, predictions)

    return _format(roc_auc_score(all_annotations, all_probs))

def ap_cc(annotations, predictions):
    """
    Calculate the average precision across all babies, by concatenating all annotations and predictions.
    :param annotations: Annotations for each baby.
    :param predictions: Predictions for each baby.
    :return: AP
    """
    all_annotations, _, all_probs = _concatenate_preds_and_annos(annotations, predictions)

    return _format(average_precision_score(all_annotations, all_probs))

def ap50_cc(annotations, predictions):
    """
    Calculate the average precision with min recall of 50 across all babies, by concatenating all annotations and predictions.
    2x the final AP value to bring it to the same scale as AP
    :param annotations: Annotations for each baby.
    :param predictions: Predictions for each baby.
    :return: AP
    """
    all_annotations, _, all_probs = _concatenate_preds_and_annos(annotations, predictions)
    min_recall = 0.5
    precision, recall, _ = precision_recall_curve(all_annotations, all_probs)

    # Ensure non-decreasing recall by reversing if necessary (should already be the case from sklearn)
    if recall[0] > recall[-1]:
        recall = recall[::-1]
        precision = precision[::-1]

    # Filter to include only the part of the curve where recall > min_recall
    valid_indices = recall >= min_recall
    recall_filtered = recall[valid_indices]
    precision_filtered = precision[valid_indices]

    # If the first recall value in the filtered set is greater than min_recall, add a point at recall = min_recall
    # by linearly interpolating precision at this recall value
    if recall_filtered[0] > min_recall:
        precision_at_min_recall = np.interp(min_recall, recall, precision)
        recall_filtered = np.insert(recall_filtered, 0, min_recall)
        precision_filtered = np.insert(precision_filtered, 0, precision_at_min_recall)

    # Append the end point (recall=1, precision=0) to close the curve, if not already present
    if recall_filtered[-1] < 1:
        recall_filtered = np.append(recall_filtered, 1)
        precision_filtered = np.append(precision_filtered, 0)

    ap = np.trapz(precision_filtered, recall_filtered)

    return _format(2*ap)

def auc_mean_std(annotations, predictions):
    """
    Calculate the mean + std AUC across babies with seizures.
    Note: this will ignore babies with no seizures or 100% seizures (where AUC is undefined)
    :param annotations: Annotations for each baby.
    :param predictions: Predictions for each baby.
    :return: mean, std AUC
    """
    auc = []
    for i, pred in enumerate(predictions):
        if sum(annotations[i]) == 0 or sum(annotations[i]) == len(annotations[i]):
            continue
        auc.append(roc_auc_score(annotations[i], pred['probs']))
    return _format(np.mean(auc)), _format(np.std(auc))

def auc_median_iqr(annotations, predictions):
    """
    Calculate the median + IQR AUC across babies with seizures.
    Note: this will ignore babies with no seizures or 100% seizures (where AUC is undefined)
    :param annotations: Annotations for each baby.
    :param predictions: Predictions for each baby.
    :return: median, IQR AUC
    """
    auc = []
    for i, pred in enumerate(predictions):
        if sum(annotations[i]) == 0 or sum(annotations[i]) == len(annotations[i]):
            continue
        auc.append(roc_auc_score(annotations[i], pred['probs']))
    return _format(np.median(auc)), _format(np.percentile(auc, 25)), _format(np.percentile(auc, 75))

def mcc(annotations, predictions):
    """
    Calculate the MCC across all babies, by concatenating all annotations and predictions.
    :param annotations: Annotations for each baby.
    :param predictions: Predictions for each baby.
    :return: MCC
    """

    all_annotations, all_preds, _ = _concatenate_preds_and_annos(annotations, predictions)

    return _format(matthews_corrcoef(all_annotations, all_preds))

def pearsons_r(annotations, predictions):
    """
    Calculate the Pearson's correlation coefficient across all babies, by concatenating all annotations and probabilities.
    :param annotations: Annotations for each baby.
    :param predictions: Predictions for each baby.
    :return: Pearson's r
    """
    all_annotations, _, all_probs = _concatenate_preds_and_annos(annotations, predictions)

    return _format(pearsonr(all_annotations, all_probs)[0])

def spearmans_r(annotations, predictions):
    """
    Calculate the Spearman's correlation coefficient across all babies, by concatenating all annotations and probabilities.
    :param annotations: Annotations for each baby.
    :param predictions: Predictions for each baby.
    :return: Spearman's r
    """
    all_annotations, _, all_probs = _concatenate_preds_and_annos(annotations, predictions)

    return _format(spearmanr(all_annotations, all_probs)[0])

def sensitivity(annotations, predictions):
    """
    Calculate the sensitivity across all babies, by concatenating all annotations and predictions.
    :param annotations: Annotations for each baby.
    :param predictions: Predictions for each baby.
    :return: sensitivity
    """
    all_annotations, all_preds, _ = _concatenate_preds_and_annos(annotations, predictions)

    return _format(recall_score(all_annotations, all_preds))

def specificity(annotations, predictions):
    """
    Calculate the specificity across all babies, by concatenating all annotations and predictions.
    :param annotations: Annotations for each baby.
    :param predictions: Predictions for each baby.
    :return: specificity
    """
    all_annotations, all_preds, _ = _concatenate_preds_and_annos(annotations, predictions)

    return _format(recall_score(all_annotations, all_preds, pos_label=0))

def ppv(annotations, predictions):
    """
    Calculate the positive predictive value (PPV) across all babies, by concatenating all annotations and predictions.
    :param annotations: Annotations for each baby.
    :param predictions: Predictions for each baby.
    :return: PPV
    """
    all_annotations, all_preds, _ = _concatenate_preds_and_annos(annotations, predictions)

    return _format(precision_score(all_annotations, all_preds))

def npv(annotations, predictions):
    """
    Calculate the negative predictive value (NPV) across all babies, by concatenating all annotations and predictions.
    :param annotations: Annotations for each baby.
    :param predictions: Predictions for each baby.
    :return: NPV
    """
    all_annotations, all_preds, _ = _concatenate_preds_and_annos(annotations, predictions)

    return _format(precision_score(all_annotations, all_preds, pos_label=0))

def _event_prediction_stats(annotation, prediction, sample_rate=1):
    """
    Calculate event based prediction masks and durations for a single baby.

    Analyses both predicted and true events and return the binary mask of overlapping events and the duration of each event (predicted and true).

    :param annotation: Annotations for a single baby.
    :param prediction: Predictions for a single baby.
    :return: predicted_events_detection, predicted_event_duration, true_events_detection, true_event_duration
    """
    def _detections(test, ref):
        # Finding continuous 1's in test
        changes = np.diff(np.concatenate(([0], test, [0])))
        start_indices = np.where(changes == 1)[0]
        end_indices = np.where(changes == -1)[0]

        # Checking each section of continuous 1's in test array for overlap with reference
        detection = []
        duration = []
        for start, end in zip(start_indices, end_indices):
            # If there's any overlap with reference, increment the good count
            if np.any(ref[start:end]):
                detection.append(1)
            else:
                detection.append(0)
            duration.append((end - start)/sample_rate)
        return np.array(detection), np.array(duration)

    predicted_events_detection, predicted_event_duration = _detections(prediction, annotation)
    true_events_detection, true_event_duration = _detections(annotation, prediction)

    return predicted_events_detection, predicted_event_duration, true_events_detection, true_event_duration

def _good_bad_missed_detections(annotation, prediction):
    """
    Calculate event based confusion matrix for a single baby.
    A good event detection is defined as a predicted seizure with _any_ overlap with an annotation.
    A bad event detection is defined as a predicted seizure without _any_ overlap with an annotation.
    A missed event detection is defined as an annotation without _any_ overlap with a prediction.

    :param annotation: Annotations for a single baby.
    :param prediction: Predictions for a single baby.
    :return: good detections, bad detections, missed detections
    """
    predicted_events_detection, _, true_events_detection, _ = _event_prediction_stats(annotation, prediction)

    good_detections = sum(predicted_events_detection)
    bad_detections = len(predicted_events_detection) - good_detections
    missed_detections = len(true_events_detection) - sum(true_events_detection)

    return good_detections, bad_detections, missed_detections

def false_event_detections_per_hour(annotations, predictions, sample_rate=1, non_seizure_cohort_only=False):
    """
    Calculate the number of false event detections across all babies.
    A false event detection is defined a predicted seizure without _any_ overlap with an annotation.

    :param annotations: Annotations for each baby.
    :param predictions: Predictions for each baby.
    :param sample_rate: Sample rate in Hz of the predictions (default is 1).
    :param non_seizure_cohort_only: If True, only calculate false event detections for babies without annotated seizures.
    :return: Number of false event detections
    """
    if non_seizure_cohort_only:
        predictions = [p for p, a in zip(predictions, annotations) if not np.any(a)]
        annotations = [a for a in annotations if not np.any(a)]

    masks = [pred['mask'] for pred in predictions]
    false_detection = 0
    for a, m in zip(annotations, masks):
        _, fp, _ = _good_bad_missed_detections(a, m)
        false_detection += fp

    total_hours = sum([len(a) for a in annotations]) / (60 * 60 * sample_rate)
    return _format(false_detection / total_hours)

def sensitivity_event(annotations, predictions):
    """
    Calculate the event based sensitivity across all babies
    A good event detection is defined as a predicted seizure with _any_ overlap with an annotation.

    :param annotations: Annotations for each baby.
    :param predictions: Predictions for each baby.
    :return: sensitivity
    """
    masks = [pred['mask'] for pred in predictions]
    tp, fn = 0, 0
    for a, m in zip(annotations, masks):
        _tp, _, _fn = _good_bad_missed_detections(a, m)
        tp += _tp
        fn += _fn
    return _format(tp / (tp + fn))

def ppv_event(annotations, predictions):
    """
    Calculate the event based positive predictive value (PPV) across all babies
    A good event detection is defined as a predicted seizure with _any_ overlap with an annotation.

    :param annotations: Annotations for each baby.
    :param predictions: Predictions for each baby.
    :return: PPV
    """
    masks = [pred['mask'] for pred in predictions]
    tp, fp, = 0, 0
    for a, m in zip(annotations, masks):
        _tp, _fp, _ = _good_bad_missed_detections(a, m)
        tp += _tp
        fp += _fp
    return _format(tp / (tp + fp))

def _per_baby_confusion_matrix(annotation, prediction):
    """
    Calculate the confusion matrix for a baby i.e.  judging only the entire recording.
    :param annotation: Annotations for a single baby.
    :param prediction: Predictions for a single baby.
    :return: TP, FP, TN, FN
    """
    label = np.max(annotation).astype(int)
    pred = np.max(prediction['mask']).astype(int)
    tp = label * pred
    fp = (1 - label) * pred
    tn = (1 - label) * (1 - pred)
    fn = label * (1 - pred)

    return tp, fp, tn, fn

def specificity_baby(annotations, predictions):
    """
    Calculate the specificity at the baby level
    i.e. judging only the entire recording which is designated a 1 if any part of the recording is a seizure and 0 otherwise.
    :param annotations: Annotations for each baby.
    :param predictions: Predictions for each baby.
    :return: specificity
    """
    tn, fp = 0, 0
    k = 0
    for a, p in zip(annotations, predictions):
        _tp, _fp, _tn, _fn = _per_baby_confusion_matrix(a, p)
        tn += _tn
        fp += _fp
        k += 1

    spec = tn / (tn + fp)
    return _format(spec)

def sensitivity_baby(annotations, predictions):
    """
    Calculate the sensitivity at the baby level
    i.e. judging only the entire recording which is designated a 1 if any part of the recording is a seizure and 0 otherwise.
    :param annotations: Annotations for each baby.
    :param predictions: Predictions for each baby.
    :return: sensitivity
    """
    tp, fn = 0, 0
    for a, p in zip(annotations, predictions):
        _tp, _fp, _tn, _fn = _per_baby_confusion_matrix(a, p)
        tp += _tp
        fn += _fn

    sens = tp / (tp + fn)
    return _format(sens)

def ppv_baby(annotations, predictions):
    """
    Calculate the PPV at the baby level
    i.e. judging only the entire recording which is designated a 1 if any part of the recording is a seizure and 0 otherwise.
    :param annotations: Annotations for each baby.
    :param predictions: Predictions for each baby.
    :return: PPV
    """
    tp, fp = 0, 0
    for a, p in zip(annotations, predictions):
        _tp, _fp, _tn, _fn  = _per_baby_confusion_matrix(a, p)
        tp += _tp
        fp += _fp

    ppv = tp / (tp + fp)
    return _format(ppv)

def npv_baby(annotations, predictions):
    """
    Calculate the NPV at the baby level
    i.e. judging only the entire recording which is designated a 1 if any part of the recording is a seizure and 0 otherwise.
    :param annotations: Annotations for each baby.
    :param predictions: Predictions for each baby.
    :return: NPV
    """
    tn, fn = 0, 0
    for a, p in zip(annotations, predictions):
        _tp, _fp, _tn, _fn  = _per_baby_confusion_matrix(a, p)
        tn += _tn
        fn += _fn

    npv = tn / (tn + fn)
    return _format(npv)

def seizure_burden(annotations, predictions, sample_rate=1, seizure_cohort_only=False):
    """
    Calculate the seizure burden on a per-baby basis.
    Seizure burden is defined as the fraction of time spent in seizure in minutes/hour.
    Here we esitimate from hourly windows

    :param annotations: Annotations for each baby.
    :param predictions: Predictions for each baby.
    :param sample_rate: Sample rate in Hz of the predictions (default is 1).
    :param seizure_cohort_only: If True, only calculate seizure burden for babies with annotated seizures.
    :return: mean predicted seizure burden, mean true seizure burden, Pearson correlation between the two
    """
    masks = [pred['mask'] for pred in predictions]
    # min/hour = (sum(mask)/(60*sample_rate)) / (len(mask)/(3600*sample_rate) = mean(mask) * 60
    if seizure_cohort_only:
        masks = [m for m, a in zip(masks, annotations) if np.any(a)]
        annotations = [a for a in annotations if np.any(a)]

    def create_60_mins_windows(arr):
        duration = int(60*60*sample_rate)
        windows = [arr[i:i+duration] for i in range(0, len(arr), duration)]
        # if < 15 mins left over, drop it
        if len(windows[-1]) < int(15*60*sample_rate) and len(windows)>1:
            windows = windows[:-1]

        return windows

    periods_annotations = []
    periods_masks = []

    for a, m in zip(annotations, masks):
        periods_annotations.extend(create_60_mins_windows(a))
        periods_masks.extend(create_60_mins_windows(m))

    true_seizure_burden = [np.mean(a) * 60 for a in periods_annotations]
    pred_seizure_burden = [np.mean(m) * 60 for m in periods_masks]

    r = np.corrcoef(true_seizure_burden, pred_seizure_burden)[0, 1]

    return true_seizure_burden, pred_seizure_burden, _format(r)

def cohens_kappa(annotations, predictions):
    """
    Calculate Cohen's kappa concatenated predictions and consensus annotations.
    :param annotations: Annotations for each baby.
    :param predictions: Predictions for each baby.
    :return: Cohen's kappa
    """
    if isinstance(predictions[0], dict):
        all_preds = np.concatenate([pred['mask'] for pred in predictions])
    else:
        all_preds = np.concatenate(predictions)
    all_annotations = np.concatenate(annotations)

    return _format(cohen_kappa_score(all_annotations, all_preds))

def cohens_kappa_pairwise(annotations, predictions):
    """
    Calculate Cohen's kappa for each annotator and the mean and std across all annotators.
    :param annotations: Annotations for each baby from each annotator.
    :param predictions: Predictions for each baby.
    :return: Mean Cohen's kappa, std Cohen's kappa
    """
    _verify_mulitple_annotators(annotations)
    cohen_kappas = []
    for annotator in range(annotations[0].shape[0]):
        annots = [a[annotator, :] for a in annotations]
        _ck = cohens_kappa(annots, predictions)
        cohen_kappas.append(_ck)
    return _format(np.mean(cohen_kappas)), _format(np.std(cohen_kappas))

def cohens_kappa_pairwise_delta(annotations, predictions):
    """
    Calculate the difference between the mean Cohen's kappa per annotator and the mean inter-annotator Cohen's kappa.
    :param annotations: Annotations for each baby from each annotator.
    :param predictions: Predictions for each baby.
    :return: Cohen's Kappa Delta
    """
    _verify_mulitple_annotators(annotations)
    cohen_kappas = []
    for i in range(annotations[0].shape[0]):
        for j in range(i+1, annotations[0].shape[0]):
            annotations_i = [a[i, :] for a in annotations]
            annotations_j = [a[j, :] for a in annotations]
            _ck = cohens_kappa(annotations_i, annotations_j)
            cohen_kappas.append(_ck)
    inter_annotator_mean = np.mean(cohen_kappas)
    ai_annotator_mean, _ = cohens_kappa_pairwise(annotations, predictions)
    delta = ai_annotator_mean - inter_annotator_mean

    return _format(delta)

def _aggregate_raters_binary(data, n_cat=2):
    '''Optimized aggregation for binary data.

    Parameters
    ----------
    data : array_like, 2-Dim
        Binary data containing category assignment with subjects in rows and
        raters in columns. Values must be 0 or 1.
    n_cat : int, optional
        Number of categories. Default is 2, which is the only valid value
        for binary data. Included for interface consistency.

    Returns
    -------
    arr : ndarray, (n_rows, n_cat)
        Contains counts of raters that assigned each category level to individuals.
        Subjects are in rows, category levels (0 and 1) in columns.
    '''
    assert n_cat == 2, 'Binary data must have 2 categories.'
    data = np.asarray(data, dtype=int)  # Ensure data is an integer array

    # Since data is binary, simply sum the ones for each row and subtract from
    # the total number of raters to get counts for zeros.
    sum_ones = data.sum(axis=1).reshape(-1, 1)
    sum_zeros = data.shape[1] - sum_ones

    # Stack the counts for zeros and ones horizontally
    counts = np.hstack((sum_zeros, sum_ones))

    cat_uni = np.array([0, 1])  # Categories are known to be 0 and 1

    return counts, cat_uni

def fleiss_kappa(annotations, predictions):
    """
    Calculate Fleiss' kappa concatenated predictions and annotations.
    :param annotations: Annotations for each baby from each annotator.
    :param predictions: Predictions for each baby.
    :return: Fleiss' kappa
    """
    _verify_mulitple_annotators(annotations)
    ai = np.concatenate([pred['mask'] for pred in predictions])
    humans = np.concatenate(annotations, axis=1)
    ratings = np.vstack((ai, humans)).T
    freq, _ = _aggregate_raters_binary(ratings)
    return _format(_fleiss_kappa(freq))

def fleiss_kappa_delta(annotations, predictions):
    """
    Calculate the difference between the Fleiss' kappa for all annotators vs the Fleiss' kappa when the AI replaces an annotator.
    :param annotations: Annotations for each baby from each annotator.
    :param annotations: Annotations for each baby from each annotator.
    :param predictions: Predictions for each baby.
    :return: Fleiss' kappa Delta
    """
    _verify_mulitple_annotators(annotations)
    ai = np.concatenate([pred['mask'] for pred in predictions])
    humans = np.concatenate(annotations, axis=1)
    human_ratings = np.stack(humans).T
    freq_human, _ = _aggregate_raters_binary(human_ratings)
    kappa_human = _fleiss_kappa(freq_human)

    kappa_ai = np.zeros(annotations[0].shape[0])
    for i in range(annotations[0].shape[0]):
        annots = humans.copy()
        annots[i] = ai
        ai_and_human_ratings = np.stack(annots).T
        freq_ai, _ = _aggregate_raters_binary(ai_and_human_ratings)
        kappa_ai_val = _fleiss_kappa(freq_ai)
        kappa_ai[i] = kappa_ai_val

    delta_mean = np.mean(kappa_ai) - kappa_human
    delta_all = [kappa - kappa_human for kappa in kappa_ai]
    return _format(delta_mean), delta_all


def fleiss_kappa_delta_bootstrap(annotations, predictions, N=1000, per_annotator=False):
    """
    Assess non-inferiority of AI using the statistical significance of Fleiss' kappa delta.
    A single bootsrap sample is drawn by resampling the annotations per baby with replacement.
    Method taken from
    "Time-Varying EEG Correlations Improve Automated Neonatal Seizure Detection" by Tapani et al.
    https://doi.org/10.1142/S0129065718500302

    :param annotations: Annotations for each baby from each annotator.
    :param annotations: Annotations for each baby from each annotator.
    :param predictions: Predictions for each baby.
    :param N: Number of bootstrap samples to draw.
    :return: p-value for AI inferiority
    """
    _verify_mulitple_annotators(annotations)
    if per_annotator:
        deltas = np.zeros((annotations[0].shape[0], N))
    else:
        deltas = np.zeros(N)
    for i in tqdm(range(N), desc='Running boostrap for non-inferiority test'):
        indices = np.random.choice(len(annotations), len(annotations), replace=True)
        sampled_annotations = [annotations[index] for index in indices]
        sampled_predictions = [predictions[index] for index in indices]
        mean_delta, all_delta = fleiss_kappa_delta(sampled_annotations, sampled_predictions)
        if per_annotator:
            deltas[:, i] = all_delta
        else:
            deltas[i] = mean_delta

    # assume deltas follow a normal distribution
    # find p-value for null hypothesis that AI is non-inferior
    if per_annotator:
        std = np.std(np.mean(deltas, axis=0))
        mean = np.mean(deltas)
        p_value_mean = 2 * min(norm.cdf(0, mean, std), 1 - norm.cdf(0, mean, std))
        print(f"{mean:.3f} ({mean - 1.96 * std:.3f}, {mean + 1.96 * std:.3f}) 95% CI, p={p_value_mean:.3f}")
    else:
        p_value_mean = 2 * min(norm.cdf(0, np.mean(deltas), np.std(deltas)), 1 - norm.cdf(0, np.mean(deltas), np.std(deltas)))
        print(f"{np.mean(deltas):.3f} ({np.mean(deltas) - 1.96 * np.std(deltas):.3f}, {np.mean(deltas) + 1.96 * np.std(deltas):.3f}) 95% CI, p={p_value_mean:.3f}")

    if per_annotator:
        stds = np.std(deltas, axis=1)
        means = np.mean(deltas, axis=1)
        p_values = [_format(2 * min(norm.cdf(0, means[i], stds[i]), 1 - norm.cdf(0, means[i], stds[i]))) for i in range(annotations[0].shape[0])]
        for i in range(annotations[0].shape[0]):
            print(f"{means[i]:.3f} ({means[i] - 1.96 * stds[i]:.3f}, {means[i] + 1.96 * stds[i]:.3f}) 95% CI, p={p_values[i]:.3f}")

        p = (p_value_mean, p_values)
    else:
        p = p_value_mean

    return p, deltas

def cross_entropy(annotations, predictions):
    """
    Calculate the cross entropy across all babies, by concatenating all annotations and predictions.
    :param annotations: Annotations for each baby.
    :param predictions: Predictions for each baby.
    :return: cross entropy
    """
    all_annotations, _, all_probs = _concatenate_preds_and_annos(annotations, predictions)

    return _format(log_loss(all_annotations, all_probs))
