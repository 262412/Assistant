import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# 初始化平滑函数
smooth_fn = SmoothingFunction().method1

# 计算 BLEU 分数
def compute_bleu(preds, labels):
    bleu_scores = []
    for pred, label in zip(preds, labels):
        pred = pred.tolist()  # 将预测结果转换为列表
        label = label.tolist()  # 将标签转换为列表

        # 计算 BLEU 分数并使用平滑函数
        bleu = sentence_bleu([label], pred, smoothing_function=smooth_fn)
        bleu_scores.append(bleu)

    return np.mean(bleu_scores)  # 返回平均 BLEU 分数

# 自定义评估指标：准确率和 BLEU 分数
def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    # 去掉填充标记 (-100 表示忽略部分)
    predictions = np.argmax(predictions, axis=-1)  # 获取预测的最大概率位置
    labels = np.where(labels != -100, labels, -1)  # 将 -100 的部分替换为 -1

    # 计算准确率
    accuracy = (predictions == labels).mean()

    # 计算 BLEU 分数
    bleu = compute_bleu(predictions, labels)

    # 输出结果，方便调试
    print(f"Accuracy: {accuracy}, BLEU: {bleu}")

    return {"accuracy": accuracy, "bleu": bleu}
