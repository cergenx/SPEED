import json

from speed.data_loaders.dataset_loader import HelsinkiDatasetLoader
from speed.data_loaders.predictions_loader import PredictionsLoader
from speed.evaluators import ConsensusEvaluator
from speed.utils import dict_to_markdown_table

# Load and parse the dataset
data_path = 'data/helsinki/annotations_2017.mat'
prediction_path = 'data/helsinki/predictions.json'

dataset_loader = HelsinkiDatasetLoader(data_path)
dataset_loader.load_dataset(consensus='unanimous') # Evaluating with unanimous consensus
annotations = dataset_loader.data['annotations']
predictions = PredictionsLoader(prediction_path, dataset_loader).load_predictions()

# Run all the evaluation based on consensus metrics
evaluator = ConsensusEvaluator(annotations, predictions)
results = evaluator.evaluate()
markdown_table = dict_to_markdown_table(results)
print("Evaluation Results:")
print(markdown_table)
