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
from torch.utils.data import DataLoader
import torch.nn.functional as F

from simulations.eenet import eenet18, eenet34, eenet50, eenet101, eenet152, eenet20, eenet32, eenet44, eenet56, eenet110
from simulations.utils import set_seed, load_config, select_optimizer, calculate_percentile_thresholds
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

        if len(costs) < len(preds):
            costs.append(torch.tensor(1.0, device=device))  # 最終コストは 1.0
        
        if len(confs) < len(preds):
            confs.append(torch.ones_like(confs[0]))

        cum_loss, pred_loss, cost_loss = loss(loss_func, None, device, preds, labels, confs, costs)

        cum_loss.backward()
        optimizer.step()


def early_test(model, test_loader, device, threshold, model_name):
    model.eval()

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    timings = []

    with torch.no_grad():
        correct_preds = 0
        total_preds = 0

        for dummy_inputs, _ in test_loader:
            dummy_inputs = dummy_inputs.to(device)
            _ = model(dummy_inputs, threshold)
            break 

        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            torch.cuda.synchronize()
            starter.record()
            pred, exit_idx, cost = model(inputs, threshold)
            
            _, outputs = torch.max(pred, dim=1)
            
            # 計測終了
            ender.record()
            torch.cuda.synchronize()

            curr_time = starter.elapsed_time(ender)
            timings.append(curr_time)

            correct_preds += outputs.eq(labels).sum().item()
            total_preds += outputs.size(0)

        acc = 100*correct_preds/total_preds
    
    return acc, timings


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))   # BranchyNet/simulations
    parent_dir = os.path.dirname(current_dir)                  # BranchyNet
    sys.path.append(parent_dir)

    config = load_config(parent_dir)

    batch_size          = config['training']['batch_size']
    epochs              = config['training']['epochs']
    lr                  = config['training']['lr']
    optimizer           = config['training']['optimizer']
    model_name          = config['training']['model_name']
    dataset             = config['training']['dataset']
    num_ee              = config['training']['num_ee']
    distribution        = config['training']['distribution']
    exit_type           = config['training']['exit_type']
    loss_func           = config['training']['loss_func']
    weight_decay        = config['training']['weight_decay']
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

    optimizer = select_optimizer(optimizer, model, lr)

    model.train()

    acc_history = []
    speed_history = []

    for ite in range(ite_num):

        print(f"iteration {ite+1}/ {ite_num}")

        set_seed(ite)

        for round in range(epochs):

            print(f"Round {round+1}/ {epochs}")

            train(optimizer, model, train_loader, device, model_name, loss_func)

        # モデルを評価
        test_thresholds = calculate_percentile_thresholds(model, val_loader, device, num_thresholds=exit_plot_num, num_ee=num_ee)
        print(f"算出された閾値: {test_thresholds}")
        for th in test_thresholds:
            test_acc, timings = early_test(model, speed_test_loader, device, th, model_name)
            speed_history.append(np.mean(timings))
            acc_history.append(test_acc)
    
    pred_speed = np.array(speed_history).reshape(ite_num, exit_plot_num+2).mean(axis=0)
    final_acc = np.array(acc_history).reshape(ite_num, exit_plot_num+2).mean(axis=0)

    # 出力先のパス設定
    save_dir = os.path.join(
        parent_dir,
        "results",
        "exp_02",
        f"{num_ee}exits_{model_name}"
        )

    os.makedirs(save_dir, exist_ok=True)

    # csv出力
    df = pd.DataFrame({
        "runtime": pred_speed,
        "accuracy": final_acc
    })
    csv_path = os.path.join(save_dir, f"exit_num={num_ee}_{model_name}.csv")
    df.to_csv(csv_path, index=False)

    #グラフ出力
    plt.figure(figsize=(10, 6))

    plt.plot(pred_speed, final_acc, marker='o', linestyle='-', color='b', label='Average vs Speed')

    # グラフのタイトルとラベル
    plt.title(f'{model_name}', fontsize=16)
    plt.xlabel('Runtime (ms)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.grid(True) # グリッド線を表示
    plt.legend() # 凡例

    # 範囲指定
    plt.xlim(0, 10)
    plt.ylim(0, 100)

    combined_path = os.path.join(save_dir, f"exit_num={num_ee}_{model_name}.png")

    plt.savefig(combined_path)
    