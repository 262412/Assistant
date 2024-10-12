import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(log_path='../logs/train_logs.npy', save_path='../result/training_metrics.png'):
    # 读取训练时记录的日志
    with open(log_path, 'rb'):
        logs = np.load(log_path, allow_pickle=True).item()
    # 检查日志内容
    print("Loaded logs:", logs)

    # 获取损失、准确率和 BLEU 分数
    # 确保这些键在日志字典中存在
    loss = logs.get('loss', None)  # 使用 get() 方法防止 KeyError
    accuracy = logs.get('eval_accuracy', None)
    bleu = logs.get('eval_bleu', None)

    # 输出检查
    print(f"Loss: {loss}, Accuracy: {accuracy}, BLEU: {bleu}")

    # 绘制图表
    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, loss, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')

    plt.subplot(1, 3, 2)
    plt.plot(epochs, accuracy, label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')

    plt.subplot(1, 3, 3)
    plt.plot(epochs, bleu, label='BLEU')
    plt.xlabel('Epoch')
    plt.ylabel('BLEU Score')
    plt.title('BLEU Score')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    plot_training_curves()
