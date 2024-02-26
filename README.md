# SPEED: Seizure Prediction Evaluation for EEG-based Detectors

## Overview
SPEED is a Python package for evaluating seizure prediction algorithms. It provides a standardized way to measure the performance of EEG-based seizure detectors using various metrics.


### Datasets

The framework only supports one dataset but is designed to be easily extended to other datasets.

| Dataset | Description | Reference |
| --- | --- | --- |
| Helsinki | 79 babies, 3 annotators | [Stevenson, N., Tapani, K., Lauronen, L. et al. A dataset of neonatal EEG recordings with seizure annotations. Sci Data 6, 190039 (2019)](https://doi.org/10.1038/sdata.2019.39) |


### Metrics

The currently supported metrics fall into three categories:

1. Sample Level Metrics - calculated by comparing prediction annotations for each sample (e.g. per second for Helsinki)
2. Event Level Metrics - calculated by comparing the prediction and annotation of contiugous seizure events using overlaps
3. Recording / Baby Level Metrics - calculated by comparing the aggregate prediction and annotation of the entire recording

An additional means of categorising the metrics is
1. Algorithim Metrics - closely follows training target; useful for comparing different algoritms; e.g. AUC, MCC
2. Agreement Metrics - measure agreement the algorithm and annotators accounting for chance; e.g. Cohen Kappa, Fleiss Kappa
3. Clinical Metrics - measures that are closer to clinical utility of the algorithm; e.g. seizure burden, False Detection Rate / Hour

For a more detailed description of each metrics, see the [Metrics](docs/metrics.md).


## Installation

```
pip install -e .
```

## Usage

To run the full evalaution suite, use the following command:

```
python -m speed.run
```
This will use the provided Helsinki annotations and some synthetic predictions.

You can customise the path to the annotations and the predictions using the following command:

```
python -m speed.run --annotations <path_to_annotations> --predictions <path_to_predictions>
```

To run the visualisation suite, use the following command:

```
python -m speed.visualise
```
which will generate some visualisations of global results by default. To change this you can supply the `--vis-type/-v` argument with `baby` or `global`. In the case of `baby` you must specify the baby to visualise using the `--baby-id/-b` argument. e.g.
```
python -m speed.visualise -v baby -b 3
```
Note: these visusalisations will look quite noisy for the provided synthetic predictions but should be more informative for real predictions.

### Using your own predictions

For now, we only support the Helsinki dataset, which is comprised of 79 babies with 3 annotators reviews at 1 second resolution. To match this, please provide a predictions file at 1 second resolution with one of the following format(s):

#### JSON file:
* a list of dictionaries, one for each baby
    * each dictionary has two keys: `mask` and `prob`
    * `mask` is a list of 0s and 1s, where 1s indicate the presence of a seizure (used for binary metrics)
    * `prob` is a list of floats, where each float is the predicted probability of a seizure at that time point (used for continuous value metrics)
    * e.g.
        ```
        [
            { "mask": [0, 0, 1, ...., 1, 0], "prob": [0.1, 0.2, 0.9, ...., 0.8, 0.1] },
            { "mask": [0, 0, 1, ...., 1, 0], "prob": [0.1, 0.2, 0.9, ...., 0.8, 0.1] },
            ...
            { "mask": [0, 0, 1, ...., 1, 0], "prob": [0.1, 0.2, 0.9, ...., 0.8, 0.1] }
        ]
        ```

### Customising the evaluation

The SPEED frameworks is designed to be flexible and customisable. You can run subsets of the evaluations or even just a single metric by importing the relevant modules and calling the functions directly.

For examples of this see `examples/` directory.

Can't find your favourite metric? Please feel free to open an issue or submit a pull request. See [Contributing](#contributing) for more details.


## Tests

To run the tests, use the following command:

```
python -m pytest
```

## Contributing

We welcome contributions from the community!

Some ideas for contributions include:
* New metrics
* New datasets
* Bug fixes
* Documentation improvements

If you'd like to improve the project, here's how you can help:

- Fork the repository: Click the 'Fork' button at the top right of this page to create your own copy of the project.
- Make your changes: Work on the improvements or fixes in your forked version.
- Submit a pull request: Once you're satisfied with your changes, open a pull request to merge your work into the main project. Please provide a brief description of your updates.

Every pull request is greatly appreciated and will be reviewed as quickly as possible. Thank you for your contribution to making this project better!


## License

This project is licensed under the MIT License - see the [LICENCE](LICENCE).