import numpy as np
from torch.utils.data import Dataset
import pickle
from utils import *

with open('../data/LINCS2020/KPGT_emb2304.pickle', 'rb') as f:
    smi2emb = pickle.load(f)

class TranSiGenDataset(Dataset):

    def __init__(self, LINCS_index, mol_feature_type, mol_id, cid, data_path='../data'):

        self.LINCS_index = LINCS_index
        self.mol_feature_type = mol_feature_type
        self.mol_id = mol_id
        self.cid = cid
        self.LINCS_data = load_from_HDF('../data/LINCS2020/processed_data.h5')
        # self.LINCS_data = load_from_HDF(data_path + '/LINCS2020/data_example/processed_data.h5')
        with open(data_path + '/LINCS2020/idx2smi.pickle', 'rb') as f:
            self.idx2smi = pickle.load(f)
        if self.mol_feature_type == 'ECFP4':
            # with open('../data/LINCS2020/ECFP4_emb2048.pickle', 'rb') as f:
            with open(data_path + '/LINCS2020/data_example/ECFP4_emb2048.pickle', 'rb') as f:
                self.smi2emb = pickle.load(f)
        elif self.mol_feature_type == 'KPGT':
            self.smi2emb = smi2emb

    def __getitem__(self, index):
        sub = subsetDict(self.LINCS_data, self.LINCS_index[index])
        mol_id = self.mol_id[index]
        smiles = self.idx2smi[mol_id]
        mol_feature = np.array(self.smi2emb[smiles], dtype=np.float32)
        return sub['x1'], sub['x2'], mol_feature, self.mol_id[index], self.cid[index], sub['sig']

    def __len__(self):
        return self.mol_id.shape[0]


class TranSiGenDataset_screening(Dataset):

    def __init__(self, x1, mol_feature, mol_id, cid):

        self.x1 = x1
        self.mol_feature = mol_feature
        self.mol_id = mol_id
        self.cid = cid

    def __getitem__(self, index):

            return self.x1[index], self.mol_feature[index], self.mol_id[index], self.cid[index]

    def __len__(self):
        return self.mol_feature.shape[0]
