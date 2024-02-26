import argparse

from speed.data_loaders.dataset_loader import HelsinkiDatasetLoader
from speed.data_loaders.predictions_loader import PredictionsLoader
from speed.visualisers import GlobalVisualiser, BabyVisualiser

def main(args):
   # Initialize DataLoader
    dataset_loader = HelsinkiDatasetLoader(args.annotations_path)

    # Load data
    dataset_loader.load_dataset(consensus='unanimous')
    predictions = PredictionsLoader(args.prediction_path , dataset_loader).load_predictions()

    if args.vis_type == 'global':
        visualiser = GlobalVisualiser(dataset_loader.data['annotations'], predictions)
        visualiser.visualise()
        return
    elif args.vis_type == 'baby':
        assert args.baby_id is not None, "Please provide baby_id for baby visualisation"
        visualiser = BabyVisualiser(dataset_loader.data['annotations'], predictions, args.baby_id)
        visualiser.visualise()
        return
    else:
        raise ValueError(f"Unknown visualiser: {args.vis}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SPEED visualisation')
    parser.add_argument('-a', '--annotations_path', type=str, default='data/helsinki/annotations_2017.mat', help='Path to the annotations')
    parser.add_argument('-p', '--prediction_path', type=str, default='data/helsinki/predictions.json', help='Path to the predictions')
    parser.add_argument('-v','--vis-type', type=str, default='global', help='Visualiser to use (global, baby)')
    parser.add_argument('-b','--baby-id', type=int, help='Baby ID for baby visualisation')

    args = parser.parse_args()
    main(args)
