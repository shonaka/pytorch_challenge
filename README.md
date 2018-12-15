# PyTorch Challenge Final Project

This is a repository for PyTorch Challenge Final Project from Udacity.
(https://github.com/udacity/pytorch_challenge)

Although the above link has a nice jupyter notebook where you could just fill in the gaps, I decided to implement the code from scratch to solve the problem for my practice.

## Data
You don't need to manually download the data because the code to handle that is already written.
If you are just interested in the data, you could download it below.
- https://s3.amazonaws.com/content.udacity-data.com/courses/nd188/flower_data.zip

## Requirements
Make sure you have Anaconda installed (or miniconda) and once you clone or download this repository, do the following.

```
conda env create -f requirements.yml
source activate pytorch_challenge
```

Current PC environment:
- RAM: 64GB
- GPUs: 1080 Ti, 1070

## Usage
For a quick experiment or testing, just do the following.
```
make -f MakeFile
```

Running experiments on your custom defined CNN
```
make -f MakeFile cnn
```

Running experiment on trasfer learned resnets
```
make -f MakeFile resnet
```

You could also modify the MakeFile itself.
Note that hyperparameter optimization using optuna or any other bayesian methods could take some time (up to a few days if you want to optimize many hyperparameters).

## Folder Structure
  ```
  pytorch-template/
  │
  ├── train.py - main script to start training
  ├── test.py - evaluation of trained model
  ├── config.json - config file
  ├── requirements.yml - requirements listed for easy construction of conda environment
  ├── cat_to_name.json - json file containing reference for the flower names
  │
  ├── data_loader/ - anything about data loading goes here
  │   └── data_loaders.py
  │
  ├── data/ - default directory for storing input data
  │
  ├── model/ - models, losses, and metrics
  │   └── model.py
  │
  ├── trainer/ - training and validating
  │   └── trainer.py
  │
  ├── results/ - default directory for storing results such as figures, checkpoints, saved models
  │
  └── utils/
      ├── util.py
      ├── logger.py - class for train logging
      ├── visualization.py - class for tensorboardX visualization support
      └── ...
  ```

## TODOs
- [x] Write requirements.yml for easy replication
- [x] Write data_loader with a3 link to download the data into the folder
- [x] Write main.py
- [x] Implement argparse with YAML
- [x] Write MakeFile for easier management and running things in batch
- [x] Write results and save them in YAML and JSON
- [x] Optimize the hyperparameters (try optuna and ray)
- [x] Implement transfer learning using pre-trained models
- [ ] Create a notebook for validation of the results.
    - [ ] Load the pre-trained network of yours
    - [ ] Visualize the example results
    - [ ] Plot confusion matrix or something

## Feedbacks
Any feedback would be greatly appreciated. Please feel free to file a github issue if you have one.

## License
This project is licensed under the MIT License. See  LICENSE for more details

## Acknowledgments
This project modified the base directory structure from [PyTorch Template Project](https://github.com/victoresque/pytorch-template) by [Victor Huang](https://github.com/victoresque)
