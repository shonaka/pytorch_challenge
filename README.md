# PyTorch Challenge Final Project

This is a repository for PyTorch Challenge Final Project from Udacity.
(https://github.com/udacity/pytorch_challenge)

Although the above link has a nice jupyter notebook where you could just fill in the gaps, I decided to implement the code from scratch to solve the problem for my practice.

## Data
You could easily download the data from below, but I wrote the data_loader script that automatically takes care of downloading and unzipping for you.
- https://s3.amazonaws.com/content.udacity-data.com/courses/nd188/flower_data.zip

## Requirements
Make sure you have Anaconda installed (or miniconda) and once you clone or download this repository, do the following.

```
conda env create -f requirements.yml
source activate pytorch_challenge
```

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
  ├── base/ - abstract base classes
  │   ├── base_data_loader.py - abstract base class for data loaders
  │   ├── base_model.py - abstract base class for models
  │   └── base_trainer.py - abstract base class for trainers
  │
  ├── data_loader/ - anything about data loading goes here
  │   └── data_loaders.py
  │
  ├── data/ - default directory for storing input data
  │
  ├── model/ - models, losses, and metrics
  │   ├── loss.py
  │   ├── metric.py
  │   └── model.py
  │
  ├── saved/ - default checkpoints folder
  │   └── runs/ - default logdir for tensorboardX
  │
  ├── trainer/ - trainers
  │   └── trainer.py
  │
  └── utils/
      ├── util.py
      ├── logger.py - class for train logging
      ├── visualization.py - class for tensorboardX visualization support
      └── ...
  ```

## TODOs
- [x] Write requirements.yml for easy replication
- [ ] Write data_loader with a3 link to download the data into the folder
- [ ] Write train.py
- [ ] Optimize the hyperparameters with optuna

## License
This project is licensed under the MIT License. See  LICENSE for more details

## Acknowledgments
This project is using the directory structure called [PyTorch Template Project](https://github.com/victoresque/pytorch-template) by [Victor Huang](https://github.com/victoresque)
