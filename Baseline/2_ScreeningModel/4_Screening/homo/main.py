import os
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import numpy as np
from tqdm import tqdm
### importing OGB
# from ogb.graphproppred import Evaluator, collate_dgl

import csv
import pandas as pd
import pickle

import sys
sys.path.append('..')

from model import Net
from utils.config import process_config, get_args
from utils.lr import MultiStepLRWarmUp
from data_preparation import MoleculeDataset


torch.set_num_threads(1)


def eval(model, device, loader):
    model.eval()
    predicted_labels = [] # 추가

    with torch.no_grad():
        for step, bg in enumerate(tqdm(loader, desc="Eval iteration")):
            bg = bg.to(device)
            x = bg.ndata.pop('feat')
            edge_attr = bg.edata.pop('feat')
            bases = bg.edata.pop('bases')
        #    labels = labels.to(device)

            pred = model(bg, x, edge_attr, bases)
            predicted_labels.extend(pred) # 추가

    return predicted_labels

def save_labels_to_csv(pred_labels, filepath): # 추가
    with open(filepath, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Index', 'Predicted Label'])
        for i in range(len(pred_labels)):
            pred_value = pred_labels[i].item()  # Extract the numerical value from the tensor
            csv_writer.writerow([i, pred_value])


import time
def main():
    args = get_args()
    config = process_config(args)
    cuda_id = os.environ.get('CUDA_VISIBLE_DEVICES')
    print(torch.cuda.get_device_name(0), cuda_id)
    print(config)

    for seed in config.seeds:
        config.seed = seed
        config.time_stamp = int(time.time())
        print(config)

        predicted_labels = run_with_given_seed(config) ######## return 수정해야 함.
        
        print('Finished label prediction')


def run_with_given_seed(config):
    if config.get('seed') is not None:
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ### automatic dataloading and splitting
    dataset = MoleculeDataset(name=config.dataset_name, config=config)

    print("Bases total: {}".format(dataset.train.graph_lists[0].edata['bases'].shape[1]))

    testset = dataset.test

    test_loader = DataLoader(testset, batch_size=config.hyperparams.batch_size, shuffle=False,
                             num_workers=config.num_workers, collate_fn=dataset.collate)

    # 인덱스 생성
    index = [i for i in range(len(dataset.test))]
    
    if config.dataset_name == 'QM':
        atom_dim = 5 # atom type 개수
        bond_dim = 4   # bond type 개수
    else:
        raise ValueError('Unknown dataset name {}'.format(config.dataset_name))
    model = Net(config.architecture, num_tasks=1,
                num_basis=dataset.test.graph_lists[0].edata['bases'].shape[1],
                shared_filter=config.get('shared_filter', '') == 'shd',
                linear_filter=config.get('linear_filter', '') == 'lin',
                atom_dim=atom_dim,
                bond_dim=bond_dim).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f'#Params: {num_params}')

    optimizer = optim.AdamW(model.parameters(), lr=config.hyperparams.learning_rate,
                            weight_decay=config.hyperparams.weight_decay)
    scheduler = MultiStepLRWarmUp(optimizer, milestones=config.hyperparams.milestones,
                                  gamma=config.hyperparams.decay_rate,
                                  num_warm_up=config.hyperparams.warmup_epochs,
                                  init_lr=config.hyperparams.learning_rate)

    
    if config.get('resume_train') is not None:
        print("Loading model from {}...".format(config.resume_train), end=' ')
        checkpoint = torch.load(config.resume_train)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        cur_epoch = checkpoint['epoch']
        #cur_loss = checkpoint['loss']
        lr = checkpoint['lr']
        print("Model loaded.")
    
        print("Epoch {} evaluating...".format(cur_epoch))
        predicted_labels = eval(model, device, test_loader)
    
    # Save predicted labels and real labels in CSV files
    save_labels_to_csv(predicted_labels, config.directory + 'Predicted_labels3.csv')  # 추가

# 최종 반환되어야 하는 것
#   1. Predictedlabels
#   2. EC HOMO 보다 높은 분자 필터링, 필터링 된 분자 개수
#   3. 필터링 된 분자의 SMILES 혹은 dgl 그래프
    predicted_labels_tensor = torch.tensor(predicted_labels)
    data = {"Index": index, 'homo': predicted_labels_tensor.cpu().numpy()}  ## index 리스트 data_preparation.py에서 뽑아와야 함.
    pred_df = pd.DataFrame(data)
    filtered_df = pred_df[pred_df['homo'] > -0.2935]
    extracted_indices = filtered_df['Index'].tolist()

    num_extracted = len(filtered_df)
    print('Number of extracted rows: ', num_extracted)

    # extracted_indices 이용해서 데이터 추출
    extracted_data = dataset.test.get_data_by_indices(extracted_indices)


    # 모든 returns 저장
    # Dataframe은 CSV로 저장
    pred_df.to_csv(os.path.join(config.directory, 'pred_df.csv'), index=False)
    filtered_df.to_csv(os.path.join(config.directory, 'filtered_df.csv'), index=False)

    # extracted indices 리스트는 text file로 저장
    with open(os.path.join(config.directory, 'extracted_indices.txt'), 'w') as f:
        for idx in extracted_indices:
            f.write(str(idx) + '\n')

    # extracted_data는 pickle file로 저장
    with open(os.path.join(config.directory, 'extracted_data.pkl'), 'wb') as f:
        pickle.dump(extracted_data, f)

    return predicted_labels , filtered_df, extracted_indices, extracted_data

if __name__ == "__main__":
    main()

