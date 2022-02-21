# ML4Science Project - Phosphorelation Site Prediction Using Deep Learning <!-- omit in toc -->

This team project is a part of the [Machine Learning](<https://www.epfl.ch/labs/mlo/machine-learning-cs-433> "EPFL's Machine Learning course") curriculum at the EPFL.

The purpose of this file is to explain the project/code structure and to help you running the scripts. 

For more information about the implementation, feel free to check out the commented code, as well as the [final report](report/Machine_Learning_Project_2.pdf), which contains the entire thought process, with all findings and conclusions.

## Table of Contents <!-- omit in toc -->

- [Abstract](#abstract)
- [Data](#data)
- [Getting Started](#getting-started)
- [Repo organization](#repo-organization)
	- [Notebooks and scripts](#notebooks-and-scripts)
- [Authors](#authors)

## Abstract

Predicting amino acid residues which are phosphorylation sites is an important problem from a biological and biomedical perspective. In this paper, we propose two neural network architectures for classifying whether residues are phosphorylation sites or not. These approaches are then compared. Additionally, we show that residue sequences have a strong representational power for the given problem. Lastly, by leveraging the outputs of the PESTO model for protein-protein interaction prediction we see that the protein-protein prediction problem is not easily transferred to phosphorylation site prediction using our methods.

## Data

For the purpose of training and model evaluation, we have used two datasets - Eukaryotic Phosphorylation Sites Database and PESTO dataset.

The necessary data can be retrieved from [here](https://drive.google.com/drive/folders/1eKOZOaClqz94sYwslzmfj9ndhhTpvpo3?usp=sharing). Note that PESTO embedding data is kept on a private drive belonging to the [EPFL's Laboratory for Biomolecular Modeling](https://www.epfl.ch/labs/lbm/). For more information, please contact us.

All of the data used in scripts and notebooks should be stored in the `data/` subdirectory.

## Getting Started

As a prerequisite, **python3** and **pip3** should already be installed.

1. Install **numpy**

  ```sh
  pip3 install numpy
  ```

2. Install **Pandas**

  ```sh
  pip3 install pandas
  ```

3. Install **matplotlib**

  ```sh
  pip3 install matplotlib
  ```

4. Install **seaborn**

  ```sh
  pip3 install seaborn
  ```

5. Install **PyTorch**

  ```sh
  pip3 install torch torchvision
  ```

6. Install **scikit-learn**

  ```sh
  pip3 install scikit-learn
  ```

7. Install **biopython**

  ```sh
  pip3 install biopython
  ```

8. Create `data/` folder inside the root directory of the repository and download the [necessary data](#data).

## Repo organization

The source code of this project is structured in the following manner:

```
project
├── README.md
│
├── data/       # data folder - needs to be added manually  
│
├── saved_models/       # contains trained models                
│
├── report/     # project report         
│    
├── analysis   # data analysis (notebook)
│   └── data_analysis.ipynb
│
└── training   # model training notebooks and scripts               
   ├── datasets.py
   ├── models.py
   ├── training_and_evaluation.py
   ├── training.ipynb
   ├── trainingCNN.ipynb
   ├── trainingLinear.ipynb
   └── utils.py

```

### Notebooks and scripts

- analysis
  - `data_analysis.ipynb` - Contains data analysis and preprocessing phase.

- training
  - `datasets.py` - Contains implementations of the Dataset classes used in training for Linear Models (_AASequenceDatasetLinear_ class) and CNN (_AASequenceDataset_).
  - `models.py` - Contains implementation of the model class used for the training of the CNN. Note that in case of the Linear model approach, models have been defined inside the notebook `trainingLinear.ipynb`.
  - `training_and_evaluation.py` - Contains implementations of the training phase for Linear model and evaluation phase for both approaches.
  - `utils.py` - Contains implementation of the util functions.
  - `trainingCNN.ipynb` - Main notebook for the training and evaluation of the CNN model.
  - `trainingLinear.ipynb` - Main notebook for the training and evaluation of the Linear model.

## Authors

- Edvin Maid
- Filip Carevic
- Natalija Mitic