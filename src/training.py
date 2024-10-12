from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from torch.utils.data import Dataset
import numpy as np
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


def train_model(save_path='../models', log_path="../logs/train_logs.npy"):
    # 初始化 GPT-2 分词器和模型
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # 添加特殊填充符号
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.resize_token_embeddings(len(tokenizer))

    # 读取并预处理文本数据
    with open("../data/preprocessed_data.txt", "r", encoding="utf-8") as f:
        text_data = f.read().splitlines()

    # 对数据进行分词处理
    tokenized_data = tokenizer(text_data, truncation=True, padding='max_length', max_length=128, return_tensors="pt")

    # 创建自定义数据集
    tokenized_dataset = CustomTextDataset(tokenized_data)

    # 设置训练参数
    training_args = TrainingArguments(
        output_dir='../models',
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
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
    train_loss = trainer.state.log_history

    # 格式化训练日志为字典
    metrics = {
        'loss': [],
        'eval_accuracy': [],
        'eval_bleu': []
    }

    # 提取损失和评估指标
    for log in train_loss:
        if 'loss' in log:
            metrics['loss'].append((log['step'], log['loss']))
        if 'eval_accuracy' in log:
            metrics['eval_accuracy'].append(log['eval_accuracy'])
        if 'eval_bleu' in log:
            metrics['eval_bleu'].append(log['eval_bleu'])

    # 使用 NumPy 保存为字典格式，避免读取问题
    np.save(log_path, metrics, allow_pickle=True)

    # 保存训练后的模型
    trainer.save_model(save_path)


if __name__ == "__main__":
    train_model()
