# Knowledge Distillation and its Effect on Subgroup Disparities for Disease Prediction
- This repository contains the code for Individual MSc Project by Jan Marczak @ Imperial College London
- The report can be viewed under `Thesis' folder

## Dataset
- The CheXpert imaging dataset together with the patient demographic information used in this work can be downloaded from https://stanfordmlgroup.github.io/competitions/chexpert/.

- The Ham10000 dataset, which contains patient information, can be accessed at https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T.

### Data Splits
- Our training splits (csv files) for all original/fair/unfair data compositions are present under `framework/data' folder.
  

## Code

For running the code, we recommend setting up a dedicated Python environment.

### Setup Python environment using conda

Create and activate a Python 3 conda environment:

````
conda create -n distillation python=3 --file requirements.txt
conda activate distillation
````

Install PyTorch using conda (for CUDA Toolkit 11.3):

````
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
````

### Results
- We provide all output predictions in `results' folder. They can be used to reproduce performance plots
- Output predictions alongisde feature embeddings can be found here [here](). These can be used to directly reproduce the plots/results by running the notebooks
- Due to the extensive sizes, the models have not been saved but can be re-trained using config files. We provide 2 main low capacity teachers under `teachers' folder. To download the high capacity ResNet101 models click [here](https://www.google.com](https://drive.google.com/drive/folders/10cqT0hQcW6s-nVRoI6XSZk5Iarx1urYl?usp=sharing)https://drive.google.com/drive/folders/10cqT0hQcW6s-nVRoI6XSZk5Iarx1urYl?usp=sharing)
