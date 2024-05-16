import json

import numpy as np

from speed.data_loaders.dataset_loader import DatasetLoader

class PredictionsLoader:
    def __init__(self, data_path, dataset_loader: DatasetLoader, sample_freq=1):
        """
        Constructor for the PredictionsLoader class.
        """
        self.data_path = data_path
        if not dataset_loader.loaded:
            raise ValueError("DatasetLoader must be loaded before initializing PredictionsLoader.")

        self.masks = dataset_loader.data['masks']
        self.raw_annotations = dataset_loader.data['raw']
        self.sample_freq = sample_freq

    def load_predictions(self, format='json'):
        """
        Load predictions from the specified file.

        :param path: String path to the file containing predictions.
        :param format: The format of the file (default is 'json').
                       Supported formats: 'json'
        :return: DataFrame containing the predictions.
        """
        if format == 'json':
            with open(self.data_path, 'rb') as f:
                predictions = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {format}")
        self._verify_predictions_shape(predictions)
        return self._process_predictions(predictions)


    def _process_predictions(self, predictions):
        """
        Apply the annotation mask to the predictions.

        :param predictions: Raw predictions.
        :return: DataFrame containing the predictions.
        """
        for i, pred in enumerate(predictions):
            predictions[i] = {'mask': np.array(pred['mask'])[self.masks[i]],
                              'probs': np.array(pred['probs'])[self.masks[i]]}

        return predictions

    def _verify_predictions_shape(self, predictions):
        """
        Verify that the predictions have the correct shape.

        :param predictions: Raw predictions.
        """
        for i, p in enumerate(predictions):
            assert isinstance(p, dict) and set(p.keys()) == {'probs', 'mask'}, f"Prediction for baby {i} is not a dict with keys 'probs' and 'mask'."
            if len(p['mask']) != self.raw_annotations[i].shape[1] or len(p['probs']) != self.raw_annotations[i].shape[1]:
                raise ValueError(f"Prediction shape ({len(p['mask'])}) does not match mask shape for baby {i} ({self.raw_annotations[i].shape[1]}).")
