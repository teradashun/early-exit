# 推論速度（バッチサイズ: 1）

import time
import copy
import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F

from src.models import DNN, MNIST_CNN, CIFAR10_CNN, LeNet_5, AlexNet
from src.BranchyNet import B_LeNet_5, B_AlexNet
from src.dataset import get_datasets, split_dataset
from src.utils import set_seed, choose_clients, select_optimizer, load_config, acc_average
from src.trainer import train, test, early_test
from src.server import federated_learning


def test(model, test_loader, device):

    # model 評価モードに設定
    model.eval()

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    timings = []

    with torch.no_grad():
        correct_preds = 0
        total_preds = 0

        for dummy_inputs, _ in test_loader:
            dummy_inputs = dummy_inputs.to(device)
            _ = model(dummy_inputs)
            break

        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 計測開始
            torch.cuda.synchronize()
            starter.record()

            outputs = model(inputs)
            
            _, preds = torch.max(outputs, 1)

            # 計測終了
            ender.record()
            torch.cuda.synchronize()

            curr_time = starter.elapsed_time(ender)
            timings.append(curr_time)
            
            correct_preds += preds.eq(labels).sum().item()
            total_preds += outputs.size(0)

        acc = 100*correct_preds/total_preds
    
    return acc, timings


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
            _ = model(dummy_inputs)
            break 

        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            if model_name == "B_LeNet_5":
                # 計測開始
                torch.cuda.synchronize()
                starter.record()

                out_branch, out_main = model(inputs)
                probs_branch = F.softmax(out_branch, dim=1)
                entropy = torch.sum(-probs_branch * torch.log(probs_branch + 1e-9), dim=1)

                _, preds_branch = torch.max(probs_branch, dim=1)
                _, preds_main = torch.max(out_main, dim=1)

                outputs = torch.where(entropy <= threshold, preds_branch, preds_main)
            
                # 計測終了
                ender.record()
                torch.cuda.synchronize()

                curr_time = starter.elapsed_time(ender)
                timings.append(curr_time)

            elif model_name == "B_AlexNet":
                # 計測開始
                torch.cuda.synchronize()
                starter.record()

                out_1, out_2, out_3 = model(inputs)

                """
                # ソフトマックスで閾値判定
                probs_branch = F.softmax(out_branch, dim=1)
                probs_main = F.softmax(out_main, dim=1)

                probs, preds_branch = torch.max(probs_branch, dim=1)
                _, preds_main = torch.max(probs_main, dim=1)
                outputs = torch.where(probs >= threshold, preds_branch, preds_main)
                """
                
                # エントロピーで閾値判定
                probs_branch1 = F.softmax(out_1, dim=1)
                entropy1 = torch.sum(-probs_branch1 * torch.log(probs_branch1 + 1e-9), dim=1)

                probs_branch2 = F.softmax(out_2, dim=1)
                entropy2 = torch.sum(-probs_branch2 * torch.log(probs_branch2 + 1e-9), dim=1)

                _, preds_branch1 = torch.max(probs_branch1, dim=1)
                _, preds_branch2 = torch.max(probs_branch2, dim=1)
                _, preds_main = torch.max(out_3, dim=1)

                # マスクの作成（閾値判定）
                exit1_mask = entropy1 <= threshold[0]
                exit2_mask = (~exit1_mask) & (entropy2 <= threshold[1])
                main_mask  = ~(exit1_mask | exit2_mask)

                # 最終予測結果の統合
                outputs = torch.zeros_like(labels)
                outputs[exit1_mask] = preds_branch1[exit1_mask]
                outputs[exit2_mask] = preds_branch2[exit2_mask]
                outputs[main_mask]  = preds_main[main_mask]

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

    batch_size    = config['training']['batch_size']
    epochs        = config['training']['epochs']
    lr            = config['training']['lr']
    optimizer     = config['training']['optimizer']
    model_name    = config['training']['model_name']
    dataset       = config['training']['dataset']
    early_exit    = config['training']['early_exit']
    threshold     = config['training']['threshold']

    global_rounds = config['FL']['global_rounds']
    ite_num       = config['FL']['ite_num']


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader, test_loader = get_datasets(batch_size, dataset)
    _, speed_test_loader = get_datasets(1, dataset)

    if model_name == "DNN":
        model = DNN().to(device)
    elif model_name == "MNIST_CNN":
        model = MNIST_CNN().to(device)
    elif model_name == "CIFAR10_CNN":
        model = CIFAR10_CNN().to(device)
    elif model_name == "LeNet_5":
        model = LeNet_5().to(device)
    elif model_name == "AlexNet":
        model = AlexNet().to(device)
    elif model_name == "B_LeNet_5":
        model = B_LeNet_5().to(device)
    elif model_name == "B_AlexNet":
        model = B_AlexNet().to(device)
    
    optimizer = select_optimizer(optimizer, model, lr)

    model.train()

    acc_history = []
    speed_history = []

    for ite in range(ite_num):

        print(f"iteration {ite+1}/ {ite_num}")

        set_seed(ite)

        for round in range(global_rounds):

            print(f"Round {round+1}/ {global_rounds}")

            train(optimizer, model, train_loader, device, model_name)

        # モデルを評価
        if model_name == "B_LeNet_5":
            test_thresholds = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, float('inf')]

            for th in test_thresholds:
                acc, timings = early_test(model, speed_test_loader, device, th, model_name)
                speed_history.append(np.mean(timings))
                acc_history.append(test_acc)

        elif model_name == "B_AlexNet":
            test_thresholds = [
                [0.01, 0.01], [0.05, 0.05], [0.1, 0.1], 
                [0.5, 0.5], [1.0, 1.0], [float('inf'), float('inf')]
            ]

            for th in test_thresholds:
                test_acc, timings = early_test(model, speed_test_loader, device, th, model_name)
                speed_history.append(np.mean(timings))
                acc_history.append(test_acc)

        else:
            test_acc, timings = test(model, speed_test_loader, device)
            speed_history.append(np.mean(timings))
            acc_history.append(test_acc)

    pred_speed = np.mean(speed_history)
    final_acc = np.mean(acc_history)

    # 出力先のパス設定
    save_dir = os.path.join(
        parent_dir,
        "results",
        "exp_01",
        f"{dataset}_"
        f"{model_name}"
        )

    os.makedirs(save_dir, exist_ok=True)

    # csv出力
    df = pd.DataFrame({
        "round": np.arange(1, global_rounds + 1),
        "accuracy": final_acc
    })
    csv_path = os.path.join(save_dir, f"{model_name}.csv")
    df.to_csv(csv_path, index=False)

    #グラフ出力
    plt.figure(figsize=(10, 6))

    rounds = np.arange(1, global_rounds + 1) # グラフのX軸（ラウンド数）

    plt.plot(rounds, final_acc, linestyle='-', color='b', label='Simple Average')

    # グラフのタイトルとラベル
    plt.title(f'{model_name}', fontsize=16)
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.grid(True) # グリッド線を表示
    plt.legend() # 凡例

    # 範囲指定
    plt.xlim(0, global_rounds)
    plt.ylim(0, 100)

    combined_path = os.path.join(save_dir, f"{model_name}.png")

    plt.savefig(combined_path)
        