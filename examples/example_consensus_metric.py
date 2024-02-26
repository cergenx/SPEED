from speed.data_loaders.dataset_loader import HelsinkiDatasetLoader
from speed.data_loaders.predictions_loader import PredictionsLoader

from speed.metrics.performance_metrics import auc_cc

# Load and parse the dataset
data_path = 'data/helsinki/annotations_2017.mat'
prediction_path = 'data/helsinki/predictions.json'

dataset_loader = HelsinkiDatasetLoader(data_path)
dataset_loader.load_dataset(consensus='unanimous') # Evaluating with unanimous consensus
annotations = dataset_loader.data['annotations']
predictions = PredictionsLoader(prediction_path, dataset_loader).load_predictions()

_auc = auc_cc(annotations, predictions)
print(f"AUC_cc: {_auc}")
