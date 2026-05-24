import os
import yaml
import random
import numpy as np
import torch
import torch.optim as optim


def load_config(dir):   #引数のパスはFedAvg
    config_path = os.path.join(dir, "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def acc_average(acc_history: list[list]) -> list:
    acc_history = np.array(acc_history)
    final_acc = np.mean(acc_history, axis=1)
    return final_acc


def select_optimizer(optim_name, local_model, lr):
    if optim_name == "SGD":
        return optim.SGD(local_model.parameters(), lr=lr, weight_decay=1e-4, momentum=0.9)
    
    elif optim_name == "Adam":
        return optim.Adam(local_model.parameters(), lr=lr)
    
    else:
        raise NameError(f"オプティマイザの名前 '{optim_name}' はサポートされていません")

"""
def choose_clients(num_clients, cohort):
    return random.sample(range(num_clients), cohort)


def acc_average(acc_history: list[list]) -> list:
    acc_history = np.array(acc_history)
    final_acc = np.mean(acc_history, axis=1)
    return final_acc
"""


def calculate_percentile_thresholds(model, val_loader, device, num_ee, num_thresholds=8):
    """
    EENetの全Exitに対して、パーセンタイルに基づいた確信度閾値を計算する
    """

    # 全Exitの出力を得るために一時的にtrainモードにする
    model.train()
    
    all_confs = [[] for _ in range(num_ee)]
    
    with torch.no_grad():
        for inputs, _ in val_loader:
            inputs = inputs.to(device)
            _, confs, _ = model(inputs)
            
            # Sigmoidから信頼度を計算
            for i in range(num_ee):
                all_confs[i].extend(confs[i].cpu().numpy().flatten())

    percentiles = np.linspace(0, 100, num_thresholds)
    
    # 各Exitごとの閾値リストを計算
    # exit_thresholds_per_stageの構造: [[Exit1_th, Exit1_th, ...], [Exit2_th, Exit2_th, ...], ...]
    exit_thresholds_per_stage = []
    for i in range(num_ee):
        th_list = np.percentile(all_confs[i], percentiles)
        exit_thresholds_per_stage.append(th_list)
    
    # exit_thresholds_per_stageを転置
    # 構造: [[Exit1_th, Exit2_th, ...], [Exit1_th, Exit2_th, ...], ...]
    thresholds = []
    for p_idx in range(num_thresholds):
        pattern = [exit_thresholds_per_stage[i][p_idx] for i in range(num_ee)]
        thresholds.append(pattern)
    
    thresholds.insert(0, [0.0] * num_ee)  # すべてexit1から退出
    thresholds.append([1.1] * num_ee)  # すべて最後のexitから退出
    
    return thresholds