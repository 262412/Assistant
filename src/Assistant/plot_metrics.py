import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(log_path='../logs/train_logs.npy', save_path='../result/training_metrics.png'):
    # 读取训练时记录的日志
    with open(log_path, 'rb') as f:
        logs = np.load(f, allow_pickle=True).item()

    # 检查日志内容
    print("Loaded logs:", logs)

    # 获取损失、准确率和 BLEU 分数
    loss = logs.get('loss', [])  # 使用 get() 方法防止 KeyError
    accuracy = logs.get('eval_accuracy', [])
    bleu = logs.get('eval_bleu', [])

    # 输出检查
    print(f"Loss: {loss}, Accuracy: {accuracy}, BLEU: {bleu}")

    # 如果 loss 是一个字典而不是包含步数和损失的列表，调整提取方式
    if isinstance(loss, list) and len(loss) > 0 and isinstance(loss[0], tuple):
        epochs = [x[0] for x in loss]
        loss_values = [x[1] for x in loss]
    else:
        # 如果没有元组结构，假设损失是一个简单的列表
        loss_values = loss
        epochs = list(range(1, len(loss_values) + 1))  # 生成相应的 epochs

    # 确保准确率和 BLEU 也有相同数量的 epochs
    if len(accuracy) == 0:
        accuracy = [0] * len(epochs)  # 如果没有准确率数据，用 0 填充
    if len(bleu) == 0:
        bleu = [0] * len(epochs)  # 如果没有 BLEU 数据，用 0 填充

    # 绘制图表
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, loss_values, label='Loss')
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
