import os
import pandas as pd
from datasets import load_dataset
from sklearn.preprocessing import MinMaxScaler


def download_data():
    # 加载开源数据集并转换为 pandas DataFrame
    dataset = load_dataset("fka/awesome-chatgpt-prompts")
    df = pd.DataFrame(dataset['train'])  # 假设使用 'train' 子集
    print("数据集下载完成！")

    # 打印数据集前几行，了解数据结构
    print("\n数据集前几行：\n", df.head())

    # 打印列名
    print("\n列名：\n", df.columns)

    # 打印每列的数据类型
    print("\n数据类型：\n", df.dtypes)

    # 打印每列的空值数量
    print("\n空值数量：\n", df.isnull().sum())

    return df


def preprocess_data(dataset):
    # 处理空值 - 根据需要填充或删除
    dataset.fillna(0, inplace=True)  # 示例：将所有空值填充为 0
    print("\n空值已处理！")

    # 转换列的数据类型 - 假设 'prompt' 是文本列
    dataset['prompt'] = dataset['prompt'].astype(str)

    # 对分类变量进行 one-hot 编码 - 假设 'act' 是分类列
    dataset = pd.get_dummies(dataset, columns=['act'], dummy_na=True)
    print("\n分类列已 one-hot 编码！")

    # 如果有数值列，可以进行归一化 - 示例中假设没有数值列，所以此步跳过
    # numeric_cols = ['numeric_column1', 'numeric_column2']
    # dataset[numeric_cols] = MinMaxScaler().fit_transform(dataset[numeric_cols])

    return dataset


def save_data(preprocessed_data, save_dir="../data"):
    # 创建保存目录（如果不存在）
    os.makedirs(save_dir, exist_ok=True)

    # 保存为 CSV 文件
    csv_path = os.path.join(save_dir, "preprocessed_data.csv")
    preprocessed_data.to_csv(csv_path, index=False)
    print(f"预处理后的数据已保存为 CSV 文件：{csv_path}")

    # 保存为文本文件（逐行保存 'prompt' 列的文本）
    txt_path = os.path.join(save_dir, "preprocessed_data.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for text in preprocessed_data['prompt']:  # 假设 'prompt' 是文本列
            f.write(text + "\n")
    print(f"预处理后的数据已保存为文本文件：{txt_path}")


if __name__ == "__main__":
    try:
        # 下载数据集并检查数据结构
        dataset = download_data()

        # 预处理数据
        preprocessed_data = preprocess_data(dataset)
        print("数据预处理完成！")

        # 保存预处理后的数据
        save_data(preprocessed_data)

    except Exception as e:
        print(f"数据处理时出错: {e}")
