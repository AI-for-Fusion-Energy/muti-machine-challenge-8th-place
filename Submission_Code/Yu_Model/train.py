#!/usr/bin/env python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from data_process import extract_data_from_directory as rd
from data_process import CustomDataset, plot_multi, read_args, current_time
from model import CNNLSTMModel, LinearModel, LSTMModel
from torch.utils.data import DataLoader, Dataset, random_split
from random import shuffle
import argparse

if __name__ == '__main__':
    args = read_args()
    print("args:", args)

    # Check for GPU availability
    device = torch.device(f"cuda:{args.idx}" if torch.cuda.is_available() else "cpu")

    data, tags = [], []
    if args.test:
        data, tags, names = rd(["./new_test1", "./new_test2", "./data5/CMod/CMod_train"], block=args.block, slope=args.slope)
    else:
        data, tags, names = rd(["./data5/HL2A",\
                                "./data5/JText",\
                                "./data5/CMod/CMod_train"], block=args.block, slope=args.slope)

    eval_data, eval_tags, eval_names = rd(["./data5/CMod/CMod_evaluate"], block=args.block, slope=args.slope)

    all_dataset = CustomDataset(data, tags, names, (args.skip1, args.skip2), args.window_size, device)
    test_dataset = CustomDataset(eval_data, eval_tags, eval_names, (args.window_size, args.window_size), args.window_size, device)

    len_all = len(all_dataset)
    len_train = int(len_all*0.9)
    len_val = len_all - len_train
    len_test = len(test_dataset)
    print(f"lenth of all data: {len_train} - {len_val} - {len_test}")
    train_dataset, val_dataset = random_split(all_dataset, [len_train, len_val])

    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size = 1, shuffle=False)

    # Initialize the model and move it to GPU
    if args.model == "cnnlstm":
        model = CNNLSTMModel(args.input_size, args.hidden_size, args.num_layers,
                             args.window_size, args.kernel_size)
    elif args.model == "linear":
        model = LinearModel(args.input_size, args.hidden_size, args.num_layers,
                            args.window_size, args.kernel_size)
    elif args.model == "lstm":
        model = LSTMModel(args.input_size, args.hidden_size, args.num_layers,
                          args.window_size, args.kernel_size)

    model = model.to(device)

    # Loss and optimizer
    criterion = nn.BCELoss()

    optimizer = optim.Adam(model.parameters(), lr=0.1)

    # Training loop
    num_epochs = args.epoch
    val_losses = []
    train_losses = []

    # Number of epochs between validation steps
    validate_every = args.validate_every

    total_steps = num_epochs * len(train_loader)+1

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps/(2*5 + 1), # 2*N + 1, N is the wave number of lr
        eta_min=0.01, # min number of the wave
        verbose=False
    )

    lrs = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            batch_data, batch_tag, batch_name = batch
            # Forward pass
            outputs = model(batch_data)
            loss = criterion(outputs, batch_tag)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            scheduler.step()
            lrs.append(scheduler.get_last_lr())

            epoch_loss += loss.item()

        train_losses.append(epoch_loss)

        # Perform validation every 'validate_every' epochs
        if (epoch + 1) % validate_every == 0:
            model_file = f"model/cuda{args.idx}/model{current_time}_epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), model_file)
            model.eval()
            val_epoch_losses = []

            for batch in val_loader:
                batch_data, batch_tag, _ = batch
                # Forward pass
                outputs = model(batch_data)
                loss = criterion(outputs, batch_tag)
                val_epoch_losses.append(loss.item())

            mean_val_loss = sum(val_epoch_losses) / len(val_epoch_losses)
            val_losses.append(mean_val_loss)

            print(f"Epo: [{epoch + 1}/{num_epochs}] - Loss: {epoch_loss:.4f} - Val: {mean_val_loss:.4f} - {model_file}")

    plot_multi(f"fig/fig{args.idx}/train_{current_time}.png",
               [train_losses,    val_losses,        lrs            ],
               ["Training Loss", "Validation Loss", "Learning Rate"])


