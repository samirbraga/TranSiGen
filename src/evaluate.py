from dataset import TranSiGenDataset
from model import TranSiGen
from utils import *
import pickle
import argparse
import warnings
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(description="Arguments for training TranSiGen")
parser.add_argument("--data_path", type=str)
parser.add_argument("--molecule_path", type=str)

parser.add_argument("--dev", type=str, default='cuda:0')
parser.add_argument("--seed", type=int, default=364039)
parser.add_argument("--molecule_feature", type=str, default='KPGT', help='molecule_feature(KPGT, ECFP4)')
parser.add_argument("--initialization_model", type=str, default='pretrain_shRNA', help='molecule_feature(pretrain_shRNA, random)')
parser.add_argument("--split_data_type", type=str, default='smiles_split', help='split_data_type(random_split, smiles_split, cell_split)')
parser.add_argument("--train_cell_count", type=str, default='None', help='if cell_split, train_cell_count=10,50,all, else None')

parser.add_argument("--n_epochs", type=int, default=300)
parser.add_argument("--n_latent", type=int, default=100)
parser.add_argument("--molecule_feature_embed_dim", nargs='+', type=int, default=[400])
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--beta", type=float, default=0.1)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--weight_decay", type=float, default=1e-5)

parser.add_argument("--train_flag", type=bool, default=False)
parser.add_argument("--eval_metric", type=bool, default=False)
parser.add_argument("--predict_profile", type=bool, default=False)

args = parser.parse_args()

random_seed = args.seed
setup_seed(random_seed)
dev = torch.device(args.dev if torch.cuda.is_available() else 'cpu')

data = load_from_HDF(args.data_path)
cell_count = len(set(data['cid']))
print('cell count:', cell_count)
# data = subsetDict(data, np.arange(10000))

with open(args.molecule_path, 'rb') as f:
    idx2smi = pickle.load(f)

print('all data:', len(data['canonical_smiles']), len(set(data['canonical_smiles'])))

train_flag = args.train_flag
eval_metric = args.eval_metric
predict_profile = args.predict_profile

init_mode = args.initialization_model
feat_type = args.molecule_feature
if feat_type == 'KPGT':
    features_dim = 2304
elif feat_type == 'ECFP4':
    features_dim = 2048
split_type = args.split_data_type
train_cell_count = args.train_cell_count
n_folds = 5

n_epochs = args.n_epochs
n_latent = args.n_latent
features_embed_dim = args.molecule_feature_embed_dim
batch_size = args.batch_size
learning_rate = args.learning_rate
beta = args.beta
dropout = args.dropout
weight_decay = args.weight_decay

# Out dir
if split_type == 'cells_split':
    local_out = '../results/trained_models_{}_cell_{}/{}/feature_{}_init_{}_{}/'.format(cell_count, split_type, random_seed, feat_type, init_mode, train_cell_count)
    print(local_out)
else:
    local_out = '../results/trained_models_{}_cell_{}/{}/feature_{}_init_{}/'.format(cell_count, split_type, random_seed, feat_type, init_mode)


if split_type in ['random_split', 'smiles_split']:
    pair, pairv, pairt = split_data(data, n_folds=n_folds, split_type=split_type, rnds=random_seed)
elif split_type == 'cells_split':
    pair, pairv, pairt = split_data_cid(data, train_cell_count=train_cell_count)
print('===============', split_type, '================')
print('train', len(set(pair['cid'])), len(pair['canonical_smiles']), len(set(pair['canonical_smiles'])), )
print('valid', len(set(pairv['cid'])), len(pairv['canonical_smiles']), len(set(pairv['canonical_smiles'])), )
print('test', len(set(pairt['cid'])), len(pairt['canonical_smiles']), len(set(pairt['canonical_smiles'])), )
    
    
filename = local_out + 'best_model.pt'
model = torch.load(filename, map_location='cpu')
model.dev = torch.device(dev)
model.to(dev)


test = TranSiGenDataset(
    LINCS_index=pairt['LINCS_index'],
    mol_feature_type=feat_type,
    mol_id=pairt['canonical_smiles'],
    cid=pairt['cid']
)


test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4, worker_init_fn=seed_worker)


save_dir = local_out + 'predict'
isExists = os.path.exists(save_dir)
print(save_dir)
if not isExists:
    os.makedirs(save_dir)
    print('Directory created successfully')
else:
    print('Directory already exists')

print('===============Evaluate model performance==============')
setup_seed(random_seed)
_, _, test_metrics_dict_ls = model.test_model(loader=test_loader, metrics_func=['pearson', 'rmse', 'precision100'])


for name, rec_dict_value in zip(['test'], [test_metrics_dict_ls]):
    df_rec = pd.DataFrame.from_dict(rec_dict_value)
    smi_ls = []
    for smi_id in df_rec['cp_id']:
        smi_ls.append(idx2smi[smi_id])
    df_rec['canonical_smiles'] = smi_ls
    print(save_dir)
    df_rec.to_csv(save_dir + '/{}_restruction_result_all_samples.csv'.format(name), index=False)