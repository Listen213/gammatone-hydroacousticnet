# analyze_dataset.py
import os
from collections import Counter
import matplotlib.pyplot as plt

def analyze_dataset(path):
    """分析数据集分布"""
    classes = os.listdir(path)
    class_counts = {}
    
    for cls in classes:
        cls_path = os.path.join(path, cls)
        if os.path.isdir(cls_path):
            count = len([f for f in os.listdir(cls_path) if f.endswith('.png')])
            class_counts[cls] = count
    
    # 打印统计
    total = sum(class_counts.values())
    print(f"总样本数: {total}")
    print("类别分布:")
    for cls, count in class_counts.items():
        print(f"  {cls}: {count} ({count/total*100:.1f}%)")
    
    # 可视化
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_counts.keys(), class_counts.values())
    plt.title('数据集类别分布')
    plt.xlabel('类别')
    plt.ylabel('样本数')
    plt.xticks(rotation=45)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=150)
    plt.show()
    
    return class_counts

# 分析训练集
print("=== 训练集分析 ===")
train_counts = analyze_dataset("/root/autodl-tmp/Project/ships_mel_dataset_split/train")

print("\n=== 测试集分析 ===")
test_counts = analyze_dataset("/root/autodl-tmp/Project/ships_mel_dataset_split/test")