# MILESTONE
This is Aiffel MILESTONE Team project
## Project : AI 기반 리튬이온배터리용 불연성 첨가제 설계 및 역합성을 통한 반응물 도출
## 소개
- 	전해질 첨가제는 음양극 표면에 보호막 형성, 과충전 방지, 전도성 향상 등 리튬이온 배터리의 수명과 안정성을 확보하기 위한 전해질 제조 과정의 필수적인 요소임.
-	기존의 trial-error 실험방식은 인적물적 자원 소모적이며 신물질 개발에 오랜 기간이 소요됨. 양자계산을 이용한 방식은 물적 자원을 소모하지 않고 기간을 단축시킬 수는 있으나, 대용량 물질 스크리닝에는 적합하지 않음. 
-	이에 AI 기법을 활용하여 자원 소모를 최소화함과 동시에 효율적으로 불연 특성을 갖는 유기물 구조체 (첨가제) 생성 모델을 설계하고자 함. 이후 역합성을 진행하여 반응물(reactant)을 도출함으로써 사람의 노력이 들어가는 부분을 최소화하고자 함.


## 전체 워크플로우 구성 및 설명
-	전체적인 워크플로우 구성은 그림1을 통해 확인 가능함.
-	SMILES code로 분자를 구현한 데이터셋을 그래프 표현으로 변환하여 생성모델에서 학습함.
-	Novelity, Validity가 1.0이 되는 지점에서 분자를 2만개 생성함. 이때 SAS(합성가능성지표)를 3.5 이상을 충족하도록 컨디션을 걸어줌.
-	Uniqueness 지표는 0.7을 기준으로 잡음
-	생성 모델(Generation Model)을 거쳐 새로운 유기분자 DB 를 생성함.

-	전처리된 DB는 스크리닝 모델(Screening Model)을 거쳐 우수 후보군(New Additives)을 선정 함.
-	스크리닝 모델은 3개의 선별자(Descriptor)로 이루어져 있음; (1) HOMO 에너지 (2) LUMO 에너지 (3) Dipole moment(쌍극자 모멘트)
-	각 선별자는 그래프 기반 GCN 모델을 이용해 미리 분자의 특성을 사전학습하고, 순차적으로 쌓아 올림으로써 스크리닝 모델을 구성함.
-	각 선별자로 모델 initialization 시에 100, 200, 300 seed에서 각각 학습하여 pretrained model 총 9개를 생성함.
-	저장된 pretrained model을 모두 불러와 앙상블 기법으로 각 선별자를 예측함
-	스크리닝 모델을 통해 최종 선별된 우수 후보군은 역합성 모델로 전달됨.
  
-	역합성 모델은 생성된 유기분자를 분해하여 반응물(Reactants)을 도출함.
-	역합성 후 도출된 반응물들은 InchI Key로 변환되어 e-molecule dataset에 동일한 분자가 있는지 체크함.
-	있다면 역합성 완료. 없다면 Recursive iteration을 돌며 추가 역합성을 진행함.
-	최대 iteration depth는 5로 지정함. 

<div align='center'>
<image src='[https://github.com/LubyJ/MILESTONE/blob/main/images/Workflow.png' width='80%' height='80%' alt='workflow'><br>
전체 워크플로우 구성
</div>

### Screening model Pretraining source code
import os
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import numpy as np
from tqdm import tqdm

import csv

import sys
sys.path.append('..')

from model import Net
from utils.config import process_config, get_args
from utils.lr import MultiStepLRWarmUp
from data_preparation import MoleculeDataset

import wandb

torch.set_num_threads(1)

def train(model, device, loader, optimizer):
    model.train()
    loss_all = 0

    for step, (bg, labels) in enumerate(tqdm(loader, desc="Train iteration")):
        bg = bg.to(device)
        x = bg.ndata.pop('feat')
        edge_attr = bg.edata.pop('feat')
        bases = bg.edata.pop('bases')
        labels = labels.to(device)

        pred = model(bg, x, edge_attr, bases)
        optimizer.zero_grad()

        loss = F.l1_loss(pred, labels)
        loss.backward()
        optimizer.step()
        loss_all = loss_all + loss.detach().item()
    return loss_all / len(loader)


def eval(model, device, loader):
    model.eval()
    total_mae = 0
    predicted_labels = [] # 추가
    real_labels = [] # 추가

    with torch.no_grad():
        for step, (bg, labels) in enumerate(tqdm(loader, desc="Eval iteration")):
            bg = bg.to(device)
            x = bg.ndata.pop('feat')
            edge_attr = bg.edata.pop('feat')
            bases = bg.edata.pop('bases')
            labels = labels.to(device)

            pred = model(bg, x, edge_attr, bases)
            mae = F.l1_loss(pred, labels).detach().item()

            total_mae += mae
            predicted_labels.extend(pred) # 추가
            real_labels.extend(labels) # 추가

        avg_mae = total_mae / (step + 1)
    return avg_mae, predicted_labels, real_labels # 추가

def save_labels_to_csv(pred_labels, real_labels, best_epoch, filepath): # 추가
    with open(filepath, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Index', 'Predicted Label', 'Real Label'])
        for i in range(len(pred_labels)):
            pred_value = pred_labels[i].item()  # Extract the numerical value from the tensor
            real_value = real_labels[i].item()
            csv_writer.writerow([i, pred_value, real_value])

        csv_writer.writerow(['Best Epoch', best_epoch])


import time
def main():
    args = get_args()
    config = process_config(args)
    cuda_id = os.environ.get('CUDA_VISIBLE_DEVICES')
    print(torch.cuda.get_device_name(0), cuda_id)
    print(config)

    wandb.init(project="PubChemQC",
               config={'commit_id': config.commit_id[0:7],
                       'cuda_id': cuda_id,
                       'shared_filter': config.get('shared_filter', ''),
                       'linear_filter': config.get('linear_filter', ''),
                       'basis': config.basis,
                       'edgehop': config.get('edgehop', ''),
                       'epsilon': config.epsilon,
                       'power': config.power,
                       'degs': config.get('degs', ''),
                       'num_layers': config.architecture.layers,
                       'hidden_units': config.architecture.hidden,
                       'learning_rate': config.hyperparams.learning_rate,
                       'warmup_epochs': config.hyperparams.warmup_epochs,
                       'milestones': config.hyperparams.milestones,
                       'decay_rate': config.hyperparams.decay_rate,
                       'weight_decay': config.hyperparams.weight_decay,
                       'batch_size': config.hyperparams.batch_size,
                       'num_workers': config.get('num_workers', 'na')},
               save_code=True,
               name='homo_seed100')


    for seed in config.seeds:
        config.seed = seed
        config.time_stamp = int(time.time())
        print(config)

        epoch_idx, valid_curve, test_curve, trainL_curve = run_with_given_seed(config)


        best_valid_epoch = np.argmin(np.array(valid_curve))
        print('Finished test_mae: {}, Validation_mae: {}, best_epoch: {}, best loss: {}'
              .format(test_curve[best_valid_epoch], valid_curve[best_valid_epoch],
                      best_valid_epoch,  min(trainL_curve)))


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

    trainset, valset, testset = dataset.train, dataset.val, dataset.test

    train_loader = DataLoader(trainset, batch_size=config.hyperparams.batch_size, shuffle=False,
                              num_workers=config.num_workers, collate_fn=dataset.collate)
    valid_loader = DataLoader(valset, batch_size=config.hyperparams.batch_size, shuffle=False,
                              num_workers=config.num_workers, collate_fn=dataset.collate)
    test_loader = DataLoader(testset, batch_size=config.hyperparams.batch_size, shuffle=False,
                             num_workers=config.num_workers, collate_fn=dataset.collate)

    if config.dataset_name == 'PubChemQC':
        atom_dim = 8 # atom type 개수
        bond_dim = 4   # bond type 개수
    else:
        raise ValueError('Unknown dataset name {}'.format(config.dataset_name))
    model = Net(config.architecture, num_tasks=1,
                num_basis=dataset.train.graph_lists[0].edata['bases'].shape[1],
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

    epoch_idx = []
    valid_curve = []
    test_curve = []
    trainL_curve = []

    cur_epoch = 0

    best_val = float('inf') # 추가
    best_val_pred_labels = None # 추가
    best_val_epoch = None # 추가

    pred_labels_at_best_epoch = None # 추가
    best_test_epoch = None

    for epoch in range(cur_epoch + 1, config.hyperparams.epochs + 1):
        lr = scheduler.optimizer.param_groups[0]['lr']
        print("Epoch {} training...".format(epoch))
        train_loss = train(model, device, train_loader, optimizer)
        scheduler.step()

        print('Evaluating...')
        valid_perf, valid_labels, valid_real_labels = eval(model, device, valid_loader) # 추가
        test_perf, test_labels, test_real_labels = eval(model, device, test_loader)  # 추가


        print('Epoch:', epoch,
              'Validation_mae:', valid_perf,
              'Test_mae:', test_perf,
              'Train loss:', train_loss,
              'lr:', lr)

        wandb.log({"epoch": epoch, "Train loss_mae": train_loss, "Val mae":valid_perf, "Test mae": test_perf, "learning rate": lr})

        epoch_idx.append(epoch)
        valid_curve.append(valid_perf)
        test_curve.append(test_perf)
        trainL_curve.append(train_loss)

        if config.get('checkpoint_dir') is not None:
            filename_header = str(config.commit_id[0:7]) + '_' \
                       + str(config.time_stamp) + '_' \
                       + str(config.dataset_name)
            if valid_perf < best_val: # valid_perf = accuracy(mae) of validation set
                best_val = valid_perf
                best_val_pred_labels = valid_labels  # mae가 가장 낮은 epoch에서 예측된 라벨을 저장
                best_val_epoch = epoch

                pred_labels_at_best_epoch = test_labels
                best_test_epoch=epoch
                filename = filename_header + 'best.tar'
            else:
                filename = filename_header + 'curr.tar'

            print("Saving model as {}...".format(filename), end=' ')
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': train_loss,
                        'lr': lr},
                       os.path.join(config.checkpoint_dir, filename))
            print("Model saved.")

    # Save predicted labels and real labels in CSV files
    save_labels_to_csv(best_val_pred_labels, valid_real_labels, best_val_epoch, config.directory + 'valid_labels.csv')  # 추가
    save_labels_to_csv(pred_labels_at_best_epoch, test_real_labels, best_test_epoch, config.directory + 'test_labels.csv')  # 추가


    return epoch_idx, valid_curve, test_curve, trainL_curve

if __name__ == "__main__":
    main()

wandb.finish()

### Screening model end2end inferance source code (ensemble method)
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
        #   labels = labels.to(device)
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

        filtered_df_lumo, extracted_indices_lumo, extracted_fmo_data = run_with_given_seed(config)
        print('Finished fmo prediction')

        return filtered_df_lumo, extracted_indices_lumo, extracted_fmo_data


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

    print("Bases total: {}".format(dataset.test.graph_lists[0].edata['bases'].shape[1]))

    testset = dataset.test

    test_loader = DataLoader(testset, batch_size=config.hyperparams.batch_size, shuffle=False,
                             num_workers=config.num_workers, collate_fn=dataset.collate)

    # 인덱스 생성
    index = [i for i in range(len(dataset.test))]

    if config.dataset_name == 'PubChemQC':
        atom_dim = 8 # atom type 개수
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

    
    homo_file_path = '../pretrained_model/1_homo/'
    homo_chk_list = [homo_file_path+'seed100/homo100.tar', homo_file_path+'seed200/homo200.tar', homo_file_path+'seed300/homo300.tar']
    df = pd.DataFrame()
    col_index = [i for i in range(0, 4399)]
    df['Index'] =  col_index
    for homo_chk in homo_chk_list:
        print("Loading model from {}...".format(homo_chk), end=' ')
        checkpoint = torch.load(homo_chk)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        cur_epoch = checkpoint['epoch']
        lr = checkpoint['lr']
        print("Model loaded.")

        print("Epoch {} evaluating...".format(cur_epoch))
        predicted_labels = eval(model, device, test_loader)
        predicted_labels_tensor = torch.tensor(predicted_labels)
        col_name = homo_chk[-11:-4]
        df[col_name] = predicted_labels_tensor.cpu().numpy()

    df['homo_mean'] = df[['homo100', 'homo200', 'homo300']].mean(axis=1)
    
###---------------------End of HOMO Prediction--------------------####

    lumo_file_path = '../pretrained_model/2_lumo/'
    lumo_chk_list = [lumo_file_path+'seed100/lumo100.tar', lumo_file_path+'seed200/lumo200.tar', lumo_file_path+'seed300/lumo300.tar']
    for lumo_chk in lumo_chk_list:
        print("Loading model from {}...".format(lumo_chk), end=' ')
        checkpoint = torch.load(lumo_chk)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        cur_epoch = checkpoint['epoch']
        lr = checkpoint['lr']
        print("Model loaded.")

        print("Epoch {} evaluating...".format(cur_epoch))
        predicted_labels = eval(model, device, test_loader)
        predicted_labels_tensor = torch.tensor(predicted_labels)
        col_name = lumo_chk[-11:-4]
        df[col_name] = predicted_labels_tensor.cpu().numpy()

    df['lumo_mean'] = df[['lumo100', 'lumo200', 'lumo300']].mean(axis=1)


###---------------------End of LUMO Prediction--------------------####

    dipole_file_path = '../pretrained_model/3_dipole/'
    dipole_chk_list = [dipole_file_path+'seed100/dipole100.tar', dipole_file_path+'seed200/dipole200.tar', dipole_file_path+'seed300/dipole300.tar']
    for dipole_chk in dipole_chk_list:
        print("Loading model from {}...".format(dipole_chk), end=' ')
        checkpoint = torch.load(dipole_chk)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        cur_epoch = checkpoint['epoch']
        lr = checkpoint['lr']
        print("Model loaded.")

        print("Epoch {} evaluating...".format(cur_epoch))
        predicted_labels = eval(model, device, test_loader)
        predicted_labels_tensor = torch.tensor(predicted_labels)
        col_name = dipole_chk[-13:-4]
        df[col_name] = predicted_labels_tensor.cpu().numpy()

    df['dipole_mean'] = df[['dipole100', 'dipole200', 'dipole300']].mean(axis=1)
    
    
    # 필터링
    filtered_df = df[(df['homo_mean'] > -5.7) & (df['lumo_mean'] < -0.5) & (df['dipole_mean'] > 4.7)]
    num_filtered_homo = len(df[df['homo_mean'] > -5.7]) # HOMO of EC : -8.022
    num_filtered_lumo = len(df[df['lumo_mean'] < -0.5])  # LUMO of EC : 1.067
    num_filtered_dipole = len(df[df['dipole_mean'] > 4.7])  # Dipole moment of EC : 5.33

    # Print the results
    print(f"Number of data points with homo_mean > -5.7: {num_filtered_homo}")
    print(f"Number of data points with lumo_mean < -0.5: {num_filtered_lumo}")
    print(f"Number of data points with dipole_mean > 4.7: {num_filtered_dipole}")
    
    extracted_indices = filtered_df['Index'].tolist()
    num_extracted_data = len(extracted_indices)
    print('Number of extracted data satisfying all descriptors: ', num_extracted_data)

    extracted_data = dataset.test.get_data_by_indices(extracted_indices)

    # Dataframe은 CSV로 저장
    df.to_csv(('../results/pred_df.csv'), index=False)
    filtered_df.to_csv(('../results/filtered_df.csv'), index=False)

    # extracted indices 리스트는 text file로 저장
    with open(('../results/extracted_indices.txt'), 'w') as f:
        for idx in extracted_indices:
            f.write(str(idx) + '\n')

    # extracted_data는 pickle file로 저장
    with open(('../results/extracted_data.pkl'), 'wb') as f:
        pickle.dump(extracted_data, f)

    return filtered_df, extracted_indices, extracted_data


if __name__ == "__main__":
    main()

