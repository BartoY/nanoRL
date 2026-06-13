import matplotlib.pyplot as plt
import matplotlib
import os
import time

matplotlib.use('Agg')

bsz = 128
num_sim = 500


def plot_learning_curves(loss_history, train_makespan, val_makespan, save_dir="/home/yifan/hang/nanoRL/draw_loss"):
    """
    绘制并保存训练过程中的Loss和Makespan曲线
    """

    epochs = range(1, len(loss_history) + 1)

    plt.figure(figsize=(12, 5))

    # --- 图 1: Loss 曲线 ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_history, 'b-', label='Training Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # --- 图 2: Makespan 曲线 ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_makespan, 'r-', label='Train (Sampling)')
    plt.plot(epochs, val_makespan, 'g-', linewidth=2, label='Val (Greedy)')
    plt.title('Average Makespan over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Makespan')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()

    timestamp = time.strftime("%m%d_%H%M", time.localtime())

    # 保存图片
    save_path = os.path.join(save_dir, f"lr_cvs_{bsz}_{num_sim}_{timestamp}.png")
    plt.savefig(save_path, dpi=300)
    print(f"训练曲线已保存至: {save_path}")
    plt.close()
    # 显示图片
    # plt.show()