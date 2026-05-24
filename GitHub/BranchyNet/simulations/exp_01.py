# 推論速度（バッチサイズ: 1）

import time
import copy
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from simulations.eenet import eenet18, eenet34, eenet50, eenet101, eenet152, eenet20, eenet32, eenet44, eenet56, eenet110
from simulations.utils import set_seed, load_config, acc_average, select_optimizer, calculate_percentile_thresholds
from simulations.loss_functions import loss
from src.server import federated_learning
from src.dataset import get_datasets, split_dataset


def train(optimizer, model, train_loader, device, model_name, loss_func):
    criterion = nn.CrossEntropyLoss()

    model.train()

    for i, (images, labels) in enumerate(train_loader):
        if model_name in ["DNN"]:
            images, labels = images.view(-1, 28*28).to(device), labels.to(device)
        
        else:
            images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()

        preds, confs, costs = model(images)

        pred_loss = loss(num_ee, preds, labels)

        pred_loss.backward()
        optimizer.step()


def early_test(model, test_loader, device, threshold, model_name):
    model.eval()

    with torch.no_grad():
        correct_preds = 0
        total_preds = 0

        for dummy_inputs, _ in test_loader:
            dummy_inputs = dummy_inputs.to(device)
            _ = model(dummy_inputs, threshold)
            break 

        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            pred, exit_idx, cost = model(inputs, threshold)
            
            _, outputs = torch.max(pred, dim=1)

            correct_preds += outputs.eq(labels).sum().item()
            total_preds += outputs.size(0)

        acc = 100*correct_preds/total_preds
    
    return acc


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))   # BranchyNet/simulations
    parent_dir = os.path.dirname(current_dir)                  # BranchyNet
    sys.path.append(parent_dir)

    config = load_config(parent_dir)

    batch_size          = config['training']['batch_size']
    epochs              = config['training']['epochs']
    lr                  = config['training']['lr']
    optim_name          = config['training']['optim_name']
    model_name          = config['training']['model_name']
    dataset             = config['training']['dataset']
    num_ee              = config['training']['num_ee']
    distribution        = config['training']['distribution']
    exit_type           = config['training']['exit_type']
    loss_func           = config['training']['loss_func']
    zero_init_residual  = config['training']['zero_init_residual']
    exit_plot_num       = config['training']['exit_plot_num']

    ite_num             = config['FL']['ite_num']


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader, val_loader, _ = get_datasets(batch_size, dataset, val_ratio=0.2)
    _, _, speed_test_loader = get_datasets(1, dataset, val_ratio=0.2)

    num_classes = 10 if dataset in ["MNIST", "CIFAR10"] else 100

    input_shape = (1, 28, 28) if dataset == "MNIST" else (3, 32, 32)

    params = {
        "num_ee": num_ee,
        "distribution": distribution,
        "exit_type": exit_type,
        "loss_func": loss_func,
        "num_classes": num_classes,
        "input_shape": input_shape,
        "zero_init_residual": zero_init_residual,
    }

    acc_history = [[] for _ in range(epochs)]

    for ite in range(ite_num):
        print(f"iteration {ite+1}/ {ite_num}")
        set_seed(ite)

        if model_name == "AlexNet":
            model = AlexNet().to(device)
        elif model_name == "eenet18":
            model = eenet18(**params).to(device)
        elif model_name == "eenet34":
            model = eenet34(**params).to(device)
        elif model_name == "eenet50":
            model = eenet50(**params).to(device)
        elif model_name == "eenet101":
            model = eenet101(**params).to(device)
        elif model_name == "eenet152":
            model = eenet152(**params).to(device)
        elif model_name == "eenet20":
            model = eenet20(**params).to(device)
        elif model_name == "eenet32":
            model = eenet32(**params).to(device)
        elif model_name == "eenet44":
            model = eenet44(**params).to(device)
        elif model_name == "eenet56":
            model = eenet56(**params).to(device)
        elif model_name == "eenet110":
            model = eenet110(**params).to(device)
        else:
            raise ValueError(f"Invalid model name: {model_name}")
        
        optimizer = select_optimizer(optim_name, model, lr)

        # 100エポックごとに学習率を0.1倍するスケジューラー
        scheduler = StepLR(optimizer, step_size=100, gamma=0.1)

        model.train()
        for round in range(epochs):

            print(f"Round {round+1}/ {epochs}")

            train(optimizer, model, train_loader, device, model_name, loss_func)

            scheduler.step()

            # モデルを評価
            test_thresholds = calculate_percentile_thresholds(model, val_loader, device, num_thresholds=exit_plot_num, num_ee=num_ee)
            test_acc = early_test(model, speed_test_loader, device, test_thresholds[-1], model_name)
            acc_history[round].append(test_acc)
            print(f"accuracy: {test_acc:.2f}%")
        
        # モデルの保存
        model_dir = os.path.join(parent_dir, "saved_models", f"{model_name}_{num_ee}_{ite+1}")
        os.makedirs(model_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(model_dir, "model.pth"))

    final_acc = acc_average(acc_history)

    # 出力先のパス設定
    save_dir = os.path.join(
        parent_dir,
        "results",
        "exp_02",
        f"{num_ee}exits_{model_name}"
        )

    os.makedirs(save_dir, exist_ok=True)

    rounds = np.arange(1, epochs + 1)

    # csv出力
    df = pd.DataFrame({
        "round": rounds,
        "accuracy": final_acc
    })
    csv_path = os.path.join(save_dir, f"exit_num={num_ee}_{model_name}.csv")
    df.to_csv(csv_path, index=False)

    #グラフ出力
    plt.figure(figsize=(10, 6))

    plt.plot(rounds, final_acc, marker='o', linestyle='-', color='b', label='Average vs round')

    # グラフのタイトルとラベル
    plt.title(f'{model_name}', fontsize=16)
    plt.xlabel('round', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.grid(True) # グリッド線を表示
    plt.legend() # 凡例

    # 範囲指定
    plt.xlim(0, epochs + 1)
    plt.ylim(0, 100)

    combined_path = os.path.join(save_dir, f"exit_num={num_ee}_{model_name}.png")

    plt.savefig(combined_path)
    