import numpy as np


def check_logs(log_path='../logs/train_logs.npy'):
    try:
        with open(log_path, 'rb'):
            logs = np.load(log_path, allow_pickle=True).item()
        print("日志内容:", logs)
    except Exception as e:
        print(f"无法加载日志文件: {e}")


if __name__ == "__main__":
    check_logs()
