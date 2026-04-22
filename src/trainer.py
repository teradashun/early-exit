import torch
import torch.nn as nn
import torch.nn.functional as F


def train(optimizer, model, train_loader, device, model_name):
    # 評価関数の定義
    criterion = nn.CrossEntropyLoss()

    # model 学習モードに設定
    model.train()

    for i, (images, labels) in enumerate(train_loader):
        if model_name in ["DNN"]:
            # viewで1次元配列に変更
            images, labels = images.view(-1, 28*28).to(device), labels.to(device)
        
        else:
            images, labels = images.to(device), labels.to(device)
        
        # 勾配をリセット
        optimizer.zero_grad()

        if model_name == "B_LeNet_5":
            # 推論
            out_branch, out_main = model(images)
            # lossを計算
            loss_b = criterion(out_branch, labels)
            loss_m = criterion(out_main, labels)
            loss = 0.3 * loss_b + 0.7 * loss_m
        
        elif model_name == "B_AlexNet":
            # 推論
            out_1, out_2, out_3 = model(images)
            # lossを計算
            loss_1 = criterion(out_1, labels)
            loss_2 = criterion(out_2, labels)
            loss_3 = criterion(out_3, labels)
            loss = 0.2 * loss_1 + 0.3 * loss_2 + 0.5 * loss_3
        
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        # 誤差逆伝播
        loss.backward()
        # パラメータ更新
        optimizer.step()


def test(model, test_loader, device):

    # model 評価モードに設定
    model.eval()

    with torch.no_grad():
        correct_preds = 0
        total_preds = 0

        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            _, preds = torch.max(outputs, 1)
            correct_preds += preds.eq(labels).sum().item()
            total_preds += outputs.size(0)

        acc = 100*correct_preds/total_preds
    
    return acc


def early_test(model, test_loader, device, threshold, model_name):
    model.eval()

    with torch.no_grad():
        correct_preds = 0
        total_preds = 0

        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            if model_name == "B_LeNet_5":
                out_branch, out_main = model(inputs)
                probs_branch = F.softmax(out_branch, dim=1)
                entropy = torch.sum(-probs_branch * torch.log(probs_branch + 1e-9), dim=1)

                _, preds_branch = torch.max(probs_branch, dim=1)
                _, preds_main = torch.max(out_main, dim=1)

                outputs = torch.where(entropy <= threshold, preds_branch, preds_main)

            elif model_name == "B_AlexNet":
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

            correct_preds += outputs.eq(labels).sum().item()
            total_preds += outputs.size(0)

        acc = 100*correct_preds/total_preds
    
    return acc