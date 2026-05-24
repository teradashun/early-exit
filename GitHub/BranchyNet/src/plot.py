import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

def plot_integrated_results_recursive(base_directory):
    # 1. ワイルドカード ** を使用し、サブフォルダ内の全CSVファイルを再帰的に探索
    search_pattern = os.path.join(base_directory, "**", "*.csv")
    csv_files = glob.glob(search_pattern, recursive=True)
    
    if not csv_files:
        print(f"Error: {base_directory} 内、およびサブディレクトリにCSVファイルが見つかりません。")
        return

    plt.figure(figsize=(12, 7))

    # 2. 各ファイルをループで読み込みプロット
    for file in csv_files:
        df = pd.read_csv(file)
        
        # ファイル名から拡張子を除外してラベル化
        label = os.path.basename(file).replace(".csv", "")
        
        if 'round' in df.columns and 'accuracy' in df.columns:
            plt.plot(df['round'], df['accuracy'], label=label, marker='o', markersize=2, alpha=0.8)
        else:
            print(f"Skip: {file} に必要なカラムがありません。")

    # 3. グラフの装飾
    plt.title('Comparison of Accuracy Across Different Models', fontsize=14)
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.ylim(72, 82)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right')
    
    # 4. 統合グラフの保存先を results ディレクトリ直下に設定
    save_path = os.path.join(base_directory, "integrated_plot.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")

# 実行例（CSVファイルが格納されているフォルダパスを指定）
# plot_integrated_results("./results/MNIST_CNN/")

plot_integrated_results_recursive("./results/")