import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(log_path='../logs/train_logs.npy', save_path='../result/training_metrics.png'):
    try:
        logs = np.load(log_path, allow_pickle=True).item()
        print("Loaded logs:", logs)
    except Exception as e:
        print(f"Error loading logs: {e}")
        return

    loss = logs.get('loss', [])
    eval_accuracy = logs.get('eval_accuracy', [])
    eval_bleu = logs.get('eval_bleu', [])

    if not loss:
        print("No loss data found!")
        return

    epochs = [x[0] for x in loss]
    loss_values = [x[1] for x in loss]

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, loss_values, marker='o', label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')

    if eval_accuracy:
        plt.subplot(1, 3, 2)
        plt.plot(range(1, len(eval_accuracy) + 1), eval_accuracy, marker='o', label='Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy')

    if eval_bleu:
        plt.subplot(1, 3, 3)
        plt.plot(range(1, len(eval_bleu) + 1), eval_bleu, marker='o', label='BLEU Score')
        plt.xlabel('Epoch')
        plt.ylabel('BLEU Score')
        plt.title('BLEU Score')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    plot_training_curves()
