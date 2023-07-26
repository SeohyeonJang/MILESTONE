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

    wandb.init(project="pdf_test",
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
               name='homo_351epochs_2')


    for seed in config.seeds:
        config.seed = seed
        config.time_stamp = int(time.time())
        print(config)

        epoch_idx, valid_curve, test_curve, trainL_curve = run_with_given_seed(config)


        best_valid_epoch = np.argmin(np.array(valid_curve))
        print('Finished test_mae: {}, Validation_mae: {}, best_epoch: {}, best loss: {}'
              .format(test_curve[best_val_epoch], valid_curve[best_val_epoch],
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

    train_loader = DataLoader(trainset, batch_size=config.hyperparams.batch_size, shuffle=True,
                              num_workers=config.num_workers, collate_fn=dataset.collate)
    valid_loader = DataLoader(valset, batch_size=config.hyperparams.batch_size, shuffle=False,
                              num_workers=config.num_workers, collate_fn=dataset.collate)
    test_loader = DataLoader(testset, batch_size=config.hyperparams.batch_size, shuffle=False,
                             num_workers=config.num_workers, collate_fn=dataset.collate)

    if config.dataset_name == 'QM':
        atom_dim = 5 # atom type 개수
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
    # if config.get('resume_train') is not None:
    #     print("Loading model from {}...".format(config.resume_train), end=' ')
    #     checkpoint = torch.load(config.resume_train)
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     model.to(device)
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    #     cur_epoch = checkpoint['epoch']
    #     cur_loss = checkpoint['loss']
    #     lr = checkpoint['lr']
    #     print("Model loaded.")
    #
    #     print("Epoch {} evaluating...".format(cur_epoch))
    #     train_perf = eval(model, device, train_loader)
    #     valid_perf = eval(model, device, valid_loader)
    #     test_perf = eval(model, device, test_loader)
    #
    #     print('Train:', train_perf,
    #           'Validation:', valid_perf,
    #           'Test:', test_perf,
    #           'Train loss:', cur_loss,
    #           'lr:', lr)
    #
    #     epoch_idx.append(cur_epoch)
    #     train_curve.append(train_perf)
    #     valid_curve.append(valid_perf)
    #     test_curve.append(test_perf)
    #     trainL_curve.append(cur_loss)
    #
    #     writer.add_scalars('traP', {ts_fk_algo_hp: train_perf}, cur_epoch)
    #     writer.add_scalars('valP', {ts_fk_algo_hp: valid_perf}, cur_epoch)
    #     writer.add_scalars('tstP', {ts_fk_algo_hp: test_perf}, cur_epoch)
    #     writer.add_scalars('traL', {ts_fk_algo_hp: cur_loss}, cur_epoch)
    #     writer.add_scalars('lr',   {ts_fk_algo_hp: lr}, cur_epoch)

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

        wandb.log({"epoch": epoch, "Train loss_mae": train_loss, "Val mae":valid_perf, "Test mae": test_perf, "learning rate": lr},
                  commit=True)

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
