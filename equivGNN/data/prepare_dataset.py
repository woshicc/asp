import torch
import torch_geometric
from ase.data import covalent_radii
from ase.neighborlist import neighbor_list
from tqdm import tqdm
import pandas as pd

default_dtype = torch.float64
torch.set_default_dtype(default_dtype)

def cov_radii(z):
    r = covalent_radii[z]*1.35
    return r

dataset_simple = []

sa_in_energy = pd.read_pickle('./simpleads_ads_energies.pickle')
sa_in_names = pd.read_pickle('./simpleads_filenames.pickle')
sa_in_atom = pd.read_pickle('./simpleads_atoms.pickle')
adsnames = [name.split('_')[0] for name in sa_in_names]
images = [atoms for atoms in sa_in_atom]
energies = [e for e in sa_in_energy]

for image, adsname, energy in tqdm(zip(images, adsnames, energies)):
    symbols = [s for s in image.symbols]
    numbers = torch.tensor(image.numbers)
    pos = torch.tensor(image.get_positions())
    lattice = torch.tensor(image.cell.array).unsqueeze(0)
    atom_tags = image.get_tags()

    # edge_src and edge_dst are the indices of the central and neighboring atom, respectively
    # edge_shift indicates whether the neighbors are in different images / copies of the unit cell
    radial_cutoff = [cov_radii(z) for z in image.numbers]
    edge_src, edge_dst, edge_shift = neighbor_list("ijS", a=image, cutoff=radial_cutoff, self_interaction=True)

    # We are computing the relative distances + unit cell shifts from periodic boundaries
    edge_batch = pos.new_zeros(pos.shape[0], dtype=torch.long)[edge_src]
    edge_vec = (pos[torch.from_numpy(edge_dst)]
                - pos[torch.from_numpy(edge_src)]
                + torch.einsum('ni,nij->nj', torch.tensor(edge_shift, dtype=default_dtype), lattice[edge_batch]))

    data = torch_geometric.data.Data(
        atom_num=numbers, symbol=symbols, energy=energy,
        pos=pos, lattice=lattice, atom_tags=atom_tags, ads=adsname,
        edge_index=torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0),
        edge_vec=edge_vec
    )

    dataset_simple.append(data)
torch.save(dataset_simple, './dataset_simpleads.pt')
