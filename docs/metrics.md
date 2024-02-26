***

# Metrics

Note: unless otherwise specified, all metrics are calculated the consensus annotations across all human annotators. There are 2 means to form consensus annotations:
* Unanimous consensus: A sample is considered a seizure if all annotators agree it is a seizure. **Areas of disagreement discarded** from consideration. This avoids penalizing the algorithm in cases where to true label is ambiguous.
* Majority consensus: A sample is considered a seizure or non-seizure based on the opinion of the majority of annotators.


## Algorithm Metrics

* **AUC (Area Under the ROC Curve):**
    * Measures the algorithm's ability to discriminate between seizure and non-seizure samples. Ranges from 0 to 1, with 1 being perfect discrimination.
    * Note: In datasets with largest class imbalance, AUC can be misleadingly high, due to true negatives dominating contribution of false positives. In such cases, it's important to consider other metrics as well.
    * Calculated in three variations:
        * **AUC<sub>cc</sub>** : Concatenating all predictions and annotations across babies.
        * **AUC mean (std)** : Mean and standard deviation of AUC across babies with seizures.
        * **AUC median (IQR)** : Median and interquartile range (IQR) of AUC across babies with seizures.

* **AP (Average Precision / Area Under Precision-Recall Curve):**
    * Similar to AUC but focuses on performance on positive class, making it more suitable for rare event detection. Calculated as the area under the Precision-Recall (PR) curve.
    * Calculated in two ways:
        * **AP<sub>cc</sub>:** Concatenating predictions and annotations across babies
        * **AP50<sub>cc</sub>:** Average precision with a minimum recall of 0.5, scaled for comparability with AP<sub>cc</sub>. Avoids low recall part of the PR curve which can be noisy, very senstiive to small changes, and also not very relevant for clinical use.

* **MCC (Matthews Correlation Coefficient):**
    * A balanced metric considering all four quadrants of the confusion matrix (True Positives, True Negatives, False Positives, False Negatives). Ranges from -1 to 1, with 1 being perfect prediction.
    * Equivalent to Pearson's r between two binary variables
      ```
      MCC = (TP * TN - FP * FN) / sqrt((TP + FP)(TP + FN)(TN + FP)(TN + FN))
      ```

* **Pearson's r:**
    * Measures the linear correlation between the algorithm's predicted probabilities and  true seizure annotations.
    * **Formula:**
      ```
      r = (Σ(x - x̄)(y - ȳ)) / sqrt(Σ(x - x̄)² Σ(y - ȳ)²)
      ```
      where:
        * x = algorithm's probability predictions
        * y = true seizure annotations
        * x̄ = mean of algorithm's predictions
        * ȳ = mean of seizure annotations

* **Spearman's r:**
    * Measures the rank correlation between algorithm predictions and true annotations (less sensitive to outliers and probability scaling than Pearson's r). Similar to Pearson's r but applied to the ranks of x and y

* **Sensitivity (Recall):**
    * Percentage of seizure samples correctly predicted as seizures.
      ```
      Sensitivity = TP / (TP + FN)
      ```

* **Specificity:**
    * Percentage of non-seizure samples correctly predicted as non-seizures.
      ```
      Specificity = TN / (TN + FP)
      ```

* **PPV (Positive Predictive Value, Precision):**
    * Percentage of seizure predictions that are truly seizures.
      ```
      PPV = TP / (TP + FP)
      ```

* **NPV (Negative Predictive Value):**
    * Percentage of non-seizure predictions that are truly non-seizures.
      ```
      NPV = TN / (TN + FN)
      ```

### Agreement Metrics

* **Cohen's Kappa:**
    * Measures agreement between the algorithm and annotators, correcting for chance agreement. Ranges from -1 to 1; higher values indicate stronger agreement.
      ```
      Kappa = (Observed Agreement - Expected Agreement from Random) / (1 - Expected Agreement from Random)
      ```
    * Metrics calculated include:
        * **Cohen's Kappa**: Kappa compute on the consensus annotations.
        * **Pairwise Cohen's Kappa**: Kappa calculated for each (algorithm, annotator) pair.
        * **Cohen's Kappa Delta**: An inferiority measure (>= 0 suggests algorithm may be non-inferior). Difference between mean pairwise Cohen's Kappa using (algorithm, annotator) pairs and mean Cohen's Kappa using (annotator, annotator) pairs.

* **Fleiss' Kappa:**
    * Generalization of Cohen's Kappa for multiple annotators.
    * Metrics calculated include:
        * **Fleiss' Kappa**: Kappa computed on the consensus annotations.
        * **Fleiss' Kappa Delta**: An inferiority measure (>= 0 suggests algorithm may be non-inferior). Difference between Fleiss' Kappa between all human annotators and the mean of all combinations which replace one annotator with the algorithm.
        * **Inferiority Test (p-value)**: A statistical test to determine if the algorithm is non-inferior to the human annotators. Uses bootsrap resampling (by baby) to estimate the distribution. The null hypothesis is that the algorithm is non-inferior to the human annotators. A p-value < 0.05 suggests the algorithm is inferior. The calculation of the p-value makes the assumption that distribution is normal, which has been a reasonable assumption in our experiments.

### Metrics important for Clinical Utility

* **False Event Detections Per Hour:**
   * Measures how often the algorithm falsely predicts a seizure event when there's none.

* **Event-based Sensitivity, PPV:**
   * Adaptations of sensitivity and PPV, calculated based on the overlap between predicted and true seizure events.

* **Baby-based Sensitivity, Specificity, PPV, NPV:**
   * Adaptations of Sensitivity, Specificity, PPV, NPV which only consider the aggregate label of "has seizures" or "no seizures" for each baby.

* **Seizure Burden:**
    * Measure of time a baby spends in a seizure state (minutes/hour)
    * Metrics calculated include:
        * **Seizure Burden r**: Pearson's r between the algorithm's predicted seizure burden and the human annotators' seizure burden on a per-baby basis.
        * **Seizure Burden r (szr only)**: Seizure burden r calculated only for babies with seizures.
