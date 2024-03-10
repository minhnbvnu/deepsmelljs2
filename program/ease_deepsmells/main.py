# lib pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

# lib default python
import os
import random
import time
import gc

# lib science math
import numpy as np

# lib sklearn
from sklearn.model_selection import StratifiedKFold, KFold

# my class
import config
import utils
import data
import train
from model import CNN_LSTM, CNN_BiLSTM, calculate_size_lstm, CNN_LSTM_CodeBERT
import time
import datetime
from config import argument

args = argument()

if __name__ == "__main__":
    pos_weight_set = [
        torch.tensor(1.0, dtype=torch.float),
        # torch.tensor(2.0, dtype=torch.float),
        torch.tensor(4.0, dtype=torch.float),
        # torch.tensor(8.0, dtype=torch.float),
        # torch.tensor(12.0, dtype=torch.float),
        # torch.tensor(32.0, dtype=torch.float),
        # torch.tensor(84.0, dtype=torch.float),
    ]

    # kernel_size_set = [3, 4, 5, 6, 7]
    kernel_size_set = [4]

    SMELL = args.smell
    now = datetime.datetime.now()

    data_path = os.path.join(os.path.join(args.data, SMELL), config.Config.DIM)
    datasets = utils.get_all_data(data_path, SMELL)

    # Sample elements randomly from given list of ids, no replacement
    train_set = data.Dataset(datasets.train_data, datasets.train_labels)
    valid_set = data.Dataset(datasets.eval_data, datasets.eval_labels)

    # Define data loaders for training and testing data in this fold
    train_loader = DataLoader(train_set, batch_size=args.train_batchsize, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=args.valid_batchsize, shuffle=True)

    print(train_set)
    length_code = len(train_set)
    print("length_code", length_code)

    for pos_weight in pos_weight_set:
        for kernel_size in kernel_size_set:
            # Calculate size LSTM - CC
            input_size_lstm = calculate_size_lstm(
                input_size=length_code, kernel_size=kernel_size
            )

            # Initialize the model, optimizer, scheduler, loss
            if args.model == "DeepSmells":
                model = CNN_LSTM(
                    kernel_size=kernel_size,
                    input_size_lstm=input_size_lstm,
                    hidden_size_lstm=args.hidden_size_lstm,
                ).to(config.Config.DEVICE)
            if args.model == "DeepSmells-BiLSTM":
                model = CNN_BiLSTM(
                    kernel_size=kernel_size,
                    input_size_lstm=input_size_lstm,
                    hidden_size_lstm=args.hidden_size_lstm,
                ).to(config.Config.DEVICE)
            if args.model == "DeepSmells-CodeBertLSTM":
                model = CNN_LSTM_CodeBERT(hidden_size=64).to(config.Config.DEVICE)
            optimizer = optim.SGD(model.parameters(), lr=args.lr)
            # step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

            train_loss_fn, valid_loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=pos_weight
            ), nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            trainer = train.Trainer(
                device=config.Config.DEVICE,
                dataloader=(train_loader, valid_loader),
                model=model,
                loss_fns=(train_loss_fn, valid_loss_fn),
                optimizer=optimizer,
                # scheduler = step_lr_scheduler,
            )

            best_precision, best_recall, best_f1, best_mcc = trainer.fit(
                epochs=args.nb_epochs,
                output_dir=args.checkpoint_dir,
                track_dir=args.tracking_dir,
                custom_name=f'Model_RQ1_{SMELL}_{args.model}_{now.strftime("%d%m%Y_%H%M")}_posweight_{pos_weight.item()}_kernel_{kernel_size}',
                threshold=args.threshold,
            )

            # del model, optimizer, train_loss_fn, valid_loss_fn, trainer, best_precision, best_recall, best_f1, best_mcc
            gc.collect()
