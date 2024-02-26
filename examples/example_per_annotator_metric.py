from speed.data_loaders.dataset_loader import HelsinkiDatasetLoader
from speed.data_loaders.predictions_loader import PredictionsLoader

from speed.metrics.performance_metrics import cohens_kappa_pairwise

# Load and parse the dataset
data_path = 'data/helsinki/annotations_2017.mat'
prediction_path = 'data/helsinki/predictions.json'

dataset_loader = HelsinkiDatasetLoader(data_path)
dataset_loader.load_dataset(consensus='all') # Evaluating with all annotators annoations
annotations = dataset_loader.data['annotations']
predictions = PredictionsLoader(prediction_path, dataset_loader).load_predictions()

mean, std = cohens_kappa_pairwise(annotations, predictions)
print(f"Pairwise Cohen's Kappa mean (std): {mean} ({std})")
