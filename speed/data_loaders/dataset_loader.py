
import scipy.io
import numpy as np

class DatasetLoader:
    def __init__(self, data_path):
        """
        Base constructor for DatasetLoader.

        :param data_path: Path to the dataset file.
        """
        self.data_path = data_path
        self.loaded = False
        self.data = None

    def load_dataset(self):
        """
        Load dataset method to be implemented in subclass.
        """
        raise NotImplementedError("This method should be implemented in subclass.")


class HelsinkiDatasetLoader(DatasetLoader):
    def load_dataset(self, consensus='unanimous'):
        """
        Load and process the Helsinki dataset.

        :param consensus: Type of consensus ('unanimous', 'majority', or 'all').
        :return: Processed dataset.
        """
        annotations = scipy.io.loadmat(self.data_path)['annotat_new'].squeeze()
        processed_annotations, masks = self._process_annotations(annotations, consensus)
        self.loaded = True
        self.data = {
            'raw': annotations,
            'annotations': processed_annotations,
            'masks': masks,
            'num_babies': annotations.shape[0],
            'num_annotators': annotations[0].shape[0]
        }

    def _process_annotations(self, annotations, consensus):
        """
        Process annotations based on consensus type.

        :param annotations: Raw annotations from the dataset.
        :param consensus: Type of consensus.
        :return: Processed annotations and a mask indicating which annotations were used.
        """
        processed_annotations = []
        annotation_masks = []
        for i in range(annotations.shape[0]):
            a, m = self._process_baby(annotations[i], consensus)
            processed_annotations.append(a)
            annotation_masks.append(m)
        return processed_annotations, annotation_masks

    def _process_baby(self, annotations, consensus):
        """
        Process annotations for a single baby.

        :param annotations: Raw annotations for a single baby.
        :param consensus: Type of consensus.

        :return: Processed annotations, and a mask indicating which annotations were used.
        """

        a_consensus = annotations.mean(axis=0)

        if consensus == 'unanimous':
            unanimity_mask = (a_consensus == 1 ) | (a_consensus == 0)
            unamimous_annotations = a_consensus[unanimity_mask]
            return unamimous_annotations, unanimity_mask

        elif consensus == 'majority':
            majority_annotations = a_consensus.round().astype(int)
            return majority_annotations, np.ones_like(majority_annotations)
        elif consensus == 'all':
            # Use all annotations
            return annotations, np.ones_like(a_consensus).astype(bool)
        else:
            raise ValueError(f"Invalid consensus type: {consensus}")
