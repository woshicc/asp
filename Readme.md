## # equivGNN demo:
equivGNN: Expected output in ./equivGNN/pre-trained/5fold/  
test: python -W ignore train_complex_5fold.py 5fold/test simpleads 5e-3 1e-5 100 0

## # Datasets:
./equivGNN/data/  
data preprocessing: prepare_dataset.ipynb

## # pre-trained models: 
./equivGNN/pre-trained/ 

## # install:
conda create -n asp python=3.9  
conda activate asp

install asp:  
git clone https://github.com/woshicc/asp.git  
pip install torch==2.1.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html  
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html  
cd path_to_asp && pip install -r requirements.txt
