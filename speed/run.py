import argparse

from speed.data_loaders.dataset_loader import HelsinkiDatasetLoader
from speed.data_loaders.predictions_loader import PredictionsLoader
from speed.evaluators import ConsensusEvaluator, AgreementEvaluator
from speed.utils import dict_to_markdown_table

def main(args):
   # Initialize DataLoader
    dataset_loader = HelsinkiDatasetLoader(args.annotations_path)

    # Load data and perform evaluation
    dataset_loader.load_dataset(consensus='unanimous')
    predictions = PredictionsLoader(args.prediction_path , dataset_loader).load_predictions()

    evaluator = ConsensusEvaluator(dataset_loader.data['annotations'], predictions)
    results_consensus = evaluator.evaluate()


    dataset_loader.load_dataset(consensus='all')
    predictions = PredictionsLoader(args.prediction_path, dataset_loader).load_predictions()
    evaluator = AgreementEvaluator(dataset_loader.data['annotations'], predictions)
    results_agreement = evaluator.evaluate()

    results = {**results_consensus, **results_agreement}
    markdown_table = dict_to_markdown_table(results)
    print("Evaluation Results:")
    print(markdown_table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SPEED evaluation')
    parser.add_argument('-a', '--annotations_path', type=str, default='data/helsinki/annotations_2017.mat', help='Path to the annotations')
    parser.add_argument('-p', '--prediction_path', type=str, default='data/helsinki/predictions.json', help='Path to the predictions')
    args = parser.parse_args()
    main(args)
