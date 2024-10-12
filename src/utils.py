import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import GPT2Tokenizer  # 确保导入 GPT-2 分词器

# 初始化平滑函数
smooth_fn = SmoothingFunction().method1


# 计算 BLEU 分数
def compute_bleu(preds, labels, tokenizer):
    bleu_scores = []
    for pred, label in zip(preds, labels):
        # 将预测结果转换为文本
        pred_text = tokenizer.decode(pred, skip_special_tokens=True).split()  # 解码预测
        label_text = tokenizer.decode(label, skip_special_tokens=True).split()  # 解码标签

        # 计算 BLEU 分数并使用平滑函数
        bleu = sentence_bleu([label_text], pred_text, smoothing_function=smooth_fn.method1)
        bleu_scores.append(bleu)

    return np.mean(bleu_scores)  # 返回平均 BLEU 分数


# 自定义评估指标：准确率和 BLEU 分数
def compute_metrics(eval_pred, tokenizer):
    predictions, labels = eval_pred

    # 去掉填充标记 (-100 表示忽略部分)
    predictions = np.argmax(predictions, axis=-1)  # 获取预测的最大概率位置
    labels = np.where(labels != -100, labels, -1)  # 将 -100 的部分替换为 -1

    # 计算准确率
    accuracy = (predictions == labels).mean()

    # 计算 BLEU 分数
    bleu = compute_bleu(predictions, labels, tokenizer)

    # 输出结果，方便调试
    print(f"Accuracy: {accuracy}, BLEU: {bleu}")

    return {"accuracy": accuracy, "bleu": bleu}

# 在你的训练代码中调用 compute_metrics 时，确保传入 tokenizer
# 例如：compute_metrics(eval_pred, tokenizer)
