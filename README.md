This repository contains the codes necessary to perform equivariant graph neural network models.

* [**equivGNN**](./equivGNN) Catalytic descriptors are crucial to accelerating catalyst design. Here, we develop an equivariant graph neural network (equivGNN) to enable
robust structure representations and achieve accurate predictions of descriptors across complex catalytic systems.

# Instalation
```bash
conda create -n asp python=3.9  
conda activate asp

git clone https://github.com/woshicc/asp.git  
pip install torch==2.1.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html  
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html  
cd path_to_asp && pip install -r requirements.txt
```


# References & Citing
If you use the equivariant graph neural network (**equivGNN**) model in your work, please cite:
```
@article{woshicc1121,
	title = {Resolving chemical-motif similarity with enhanced atomic structure representations for accurately predicting descriptors at metallic interfaces},
	doi = {10.1038/s41467-025-63860-x},
	journaltitle = {Nature Communications},
	author = {Cai, Cheng and Wang, Tao},
	date = {2025-09},
}
```
