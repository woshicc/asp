from models.egnn_v5 import equivGNN
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from sklearn.model_selection import StratifiedKFold
import torch
import os
import logging
from sys import argv
from tqdm import tqdm

default_dtype = torch.float64
torch.set_default_dtype(default_dtype)
torch.manual_seed(5757)

batch_size = 8
dir, adsorbate = argv[1], argv[2]
ini_lr, wd, epochs = float(argv[3]), float(argv[4]), int(argv[5])
target, t_k = 'energy', int(argv[6])            # k in [0,1,2,3,4]
save_name = adsorbate+'_'+str(t_k)
assert adsorbate in ['simpleads', 'complex']

# logging
work_dir = './pre-trained/'+dir
if os.path.exists(work_dir):
    print(f'{work_dir} exist')
else:
    os.mkdir(work_dir)
    print(f'{work_dir} maked')
logger = logging.getLogger()
logger.setLevel(level=logging.INFO)
fHandler = logging.FileHandler(f'./{work_dir}/{save_name}.log', mode='w')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fHandler.setFormatter(formatter)
logger.addHandler(fHandler)
logger.info(f'workdir: {work_dir}, adsorbate: {adsorbate}')

### data loading & one-hot atom encoding ###
data = torch.load(f'./data/dataset_{adsorbate}.pt')

import json
with open('./data/atom_init.json', 'r') as f:
    key = json.load(f)
for d in tqdm(data):
    d['x'] = torch.tensor([key[str(int(i))] for i in d['atom_num']], dtype=default_dtype)           # cgcnn features: 92
    d['n'] = torch.tensor([key[str(int(i))][:26] for i in d['atom_num']], dtype=default_dtype)      # G-P attributes: 26

# dataLoader
dataset = [Data(x=d['x'], n=d['n'], ads=d['ads'], y=d[target],
                edge_index=d['edge_index'], edge_vec=d['edge_vec']) for d in data]
dataset, _, _ = random_split(dataset, [len(dataset), 0, 0], generator=torch.Generator())

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=5757)
skfold_val = {}
for i, (train_idx, val_idx) in enumerate(skf.split(dataset, [d['ads'] for d in dataset])):
    skfold_val[i] = val_idx

VALID_FOLD_INDEX = t_k
valid_idx = skfold_val[VALID_FOLD_INDEX]
valid_dataset = [dataset[i] for i in valid_idx]

train_folds = list(range(n_splits))
train_folds.remove(VALID_FOLD_INDEX)
train_idx = [i for fold in train_folds for i in skfold_val[fold]]
train_dataset = [dataset[i] for  i in train_idx]

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
### data loading & one-hot atom encoding ###

logger.info(f'dataset size: {len(dataset)}, batch size: {batch_size}')
logger.info(f'train/valid/test size: {len(train_dataset)}/{len(valid_dataset)}/0')

### e3nn net ###
net = equivGNN(irreps_node_inputs='92x0e',irreps_node_attr='26x0e')
device = "cuda:0" if torch.cuda.is_available() else "cpu"
optimizer = torch.optim.AdamW(net.parameters(), lr=ini_lr, weight_decay=wd, foreach=False)
lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, epochs=epochs,
                                               steps_per_epoch=len(train_loader),
                                               max_lr=ini_lr,
                                               )
criterion = torch.nn.MSELoss().to(device)
net.to(device)
logger.info(net)
# initial info
with torch.no_grad():
    net.eval()
    delta = 0
    for batch in val_loader:
        out = net(batch.to(device))
        delta += torch.sum(torch.abs(out-batch.y.to(device)))
logger.info(f"initial lr: {lr_scheduler.get_last_lr()[0]:.9f}, meanAE: {delta/len(valid_dataset)}")

# training
min_mae, bestModel_epoch = float('inf'), 0
for epoch in range(epochs):
    net.train()
    train_loss = 0.0
    with tqdm(train_loader) as tbar:
        for batch in tbar:
            optimizer.zero_grad()
            out = net(batch.to(device))
            loss = criterion(out, batch.y.to(device))
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            tbar.set_postfix(job=f'[Epoch {epoch+1}/{epochs}]: loss: {loss.item():.6f}, lr: {lr_scheduler._last_lr[0]:.9f}')
            lr_scheduler.step()
        train_loss = train_loss/len(train_loader)

    net.eval()
    val_mae, val_loss = 0.0, 0.0
    with torch.no_grad():
        for batch in val_loader:
            out = net(batch.to(device))
            val_loss += criterion(out, batch.y.to(device))
            val_mae += torch.sum(torch.abs(out-batch.y.to(device)))
        val_loss = val_loss/len(val_loader)
        val_mae = val_mae/len(valid_dataset)
    logger.info(f"Epoch {epoch+1}, train_loss: {train_loss:.6f}, val_loss: {val_loss:.6f}, val_mae: {val_mae:.6f}")

    if val_mae < min_mae and epoch+1 > epochs/2:
        min_mae = val_mae
        torch.save(net, f'./{work_dir}/best_model-{save_name}.pth')
        bestModel_epoch = epoch+1

torch.save(net, f'./{work_dir}/last_model-{save_name}.pth')
### e3nn net ###

model = torch.load(f'./{work_dir}/best_model-{save_name}.pth')
model.eval()
test_mae = 0.0
with torch.no_grad():
    for batch in val_loader:
        out = model(batch.to(device))
        test_mae += torch.sum(torch.abs(out-batch.y.to(device)))
    test_mae = test_mae/len(valid_dataset)
logger.info(f"Test MAE: {test_mae:.6f} with best model at Epoch {bestModel_epoch}")
