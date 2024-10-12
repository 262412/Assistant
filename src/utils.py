import numpy as np
from nltk.translate.bleu_score import sentence_bleu


# 计算 BLEU 分数
def compute_bleu(preds, labels):
    bleu_scores = []
    for pred, label in zip(preds, labels):
        pred = pred.tolist()
        label = label.tolist()
        bleu_scores.append(sentence_bleu([label], pred))  # 单个句子的 BLEU 分数
    return np.mean(bleu_scores)


# 自定义评估指标：准确率和 BLEU 分数
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # 去掉填充标记
    predictions = np.argmax(predictions, axis=-1)
    labels = np.where(labels != -100, labels, -1)  # 忽略填充部分

    accuracy = (predictions == labels).mean()  # 计算准确率
    bleu = compute_bleu(predictions, labels)  # 计算 BLEU 分数

    return {"accuracy": accuracy, "bleu": bleu}
