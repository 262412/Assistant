from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from torch.utils.data import Dataset
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from utils import compute_metrics  # 从 utils.py 导入 BLEU 和准确率计算函数


# 自定义数据集类
class CustomTextDataset(Dataset):
    def __init__(self, tokenized_data):
        self.input_ids = tokenized_data['input_ids']

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'labels': self.input_ids[idx]
        }


def compute_bleu(reference, hypothesis):
    """使用 SmoothingFunction 计算 BLEU 分数"""
    smooth_fn = SmoothingFunction().method1
    return sentence_bleu(reference, hypothesis, smoothing_function=smooth_fn)


def train_model(
    data_path='../data/preprocessed_data.txt',
    model_path='../models',
    log_path='../logs/train_logs.npy'
):
    # 初始化 GPT-2 分词器和模型
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # 添加特殊填充符号
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.resize_token_embeddings(len(tokenizer))

    # 读取并预处理文本数据
    with open(data_path, "r", encoding="utf-8") as f:
        text_data = f.read().splitlines()

    # 对数据进行分词处理
    tokenized_data = tokenizer(text_data, truncation=True, padding='max_length', max_length=128, return_tensors="pt")

    # 创建自定义数据集
    tokenized_dataset = CustomTextDataset(tokenized_data)

    # 设置训练参数
    training_args = TrainingArguments(
        output_dir=model_path,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='../logs',
        logging_steps=10,
        save_strategy="no",
        report_to='none',
        evaluation_strategy="epoch",
        fp16=True
    )

    # 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset,
        compute_metrics=compute_metrics
    )

    # 开始训练
    trainer.train()

    # 获取训练过程中的损失和日志
    train_logs = trainer.state.log_history

    # 格式化日志并计算 BLEU 分数
    metrics = {
        'loss': [],
        'eval_accuracy': [],
        'eval_bleu': []
    }

    for log in train_logs:
        if 'loss' in log:
            metrics['loss'].append(log['loss'])
        if 'eval_accuracy' in log:
            metrics['eval_accuracy'].append(log['eval_accuracy'])
        if 'eval_preds' in log and 'eval_labels' in log:
            # 计算 BLEU 分数
            hypothesis = tokenizer.decode(log['eval_preds'], skip_special_tokens=True).split()
            reference = [tokenizer.decode(log['eval_labels'], skip_special_tokens=True).split()]
            bleu_score = compute_bleu(reference, hypothesis)
            metrics['eval_bleu'].append(bleu_score)

    # 保存训练日志
    np.save(log_path, metrics, allow_pickle=True)

    # 保存训练后的模型
    trainer.save_model(model_path)


if __name__ == "__main__":
    train_model()
