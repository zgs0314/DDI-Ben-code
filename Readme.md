# Code for Benchmarking Computational Methods for Drug-Drug Interaction Prediction: A Perspective from Distribution Changes


## Installation
### 1. Set Up Environment for `DDI_ben` and `TextDDI`
The models MLP, MSTE, ComplEx, Decagon, TIGER in `DDI_ben` and `TextDDI` share the same environment. Our running environment is a Linux server with Ubuntu. You can set up the environment as follows:

```bash
# Create and activate Conda environment
conda create -n ddibench python=3.8.0
conda activate ddibench

# Install dependencies
# We provide the exact versions of packages we use
pip install -r DDI_ben/requirements.txt
```

### 2. Set Up Environment for `EmerGNN` and `SumGNN`
`EmerGNN` and `SumGNN` require different environments. Each should be set up separately according to their respective official repositories. You can find the official repositories here:  
- **EmerGNN**: [EmerGNN Repository](https://github.com/LARS-research/EmerGNN)  
- **SumGNN**: [SumGNN Repository](https://github.com/yueyu1030/SumGNN)

## Running the Code
### 1. Dataset
We provide the dataset in the following folders:
- `DDI_ben`: [DDI_ben/data](https://github.com/zgs0314/DDI-Ben-code/tree/main/DDI_Ben/DDI_ben/data)
- `TextDDI`: [TextDDI/data](https://github.com/zgs0314/DDI-Ben-code/tree/main/DDI_Ben/TextDDI/data)
- `EmerGNN`: [EmerGNN/DrugBank/data](https://github.com/zgs0314/DDI-Ben-code/tree/main/DDI_Ben/EmerGNN/DrugBank/data)
- `SumGNN`: [SumGNN/data](https://github.com/zgs0314/DDI-Ben-code/tree/main/DDI_Ben/SumGNN/data)

### 2. Running Scripts
First, `cd` into the corresponding directory, i.e., DDI_ben, TextDDI, EmerGNN/Drugbank, EmerGNN/TWOSIDES or SumGNN. After that,

- For `DDI_ben`, you can run the code as follows:
```bash
python main.py --model MSTE  --dataset drugbank --dataset_type finger --gamma_split 55  --lr 3e-3 --gpu 0 
```

- For `TextDDI`, 
```bash
python drugbank/main_drugbank.py --dataset_type finger --gamma_split 55
```

- For `EmerGNN`,
```bash
python -W ignore evaluate.py --dataset=S0_finger_55 --n_epoch=40 --epoch_per_test=2 --gpu=0
```

- For `SumGNN`,
```bash
python train.py -d drugbank_finger_55 -e drugbank_finger_55 --gpu=0 --hop=2 --batch=64 --l2=1e-5 --emb_dim=64 -b=8 --lr=5e-4 -s S0 -ne=40 --max_links 120000 -max_h 100
```