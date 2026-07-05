#!/usr/bin/env python3
"""
直接测试船舶音频分类模型
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np
import os
import sys

# ========== 配置 ==========
MODEL_PATH = "/root/autodl-tmp/Test/ConvNeXt-main/outputs/checkpoint-best.pth"
TEST_DIR = "/root/autodl-tmp/Test/ships_gammatone_final_test_set"

# 类别名称（按文件夹顺序）
CLASS_NAMES = [
    'B_Small_Speedboat',
    'C_Fishing_Boat', 
    'A_Large_Merchant_Ship',
    'D_Ferry',
    'E_Ambient_Noise'
]

# ========== 加载模型 ==========
print("🔧 加载模型...")
try:
    # 尝试不同的加载方式
    for weights_only in [False, True]:
        try:
            checkpoint = torch.load(MODEL_PATH, 
                                  map_location='cuda' if torch.cuda.is_available() else 'cpu',
                                  weights_only=weights_only)
            print(f"✅ 使用 weights_only={weights_only} 加载成功")
            break
        except:
            continue
    else:
        # 如果都失败，尝试添加安全globals
        import torch.serialization
        import numpy
        torch.serialization.add_safe_globals([numpy._core.multiarray.scalar])
        checkpoint = torch.load(MODEL_PATH,
                              map_location='cuda' if torch.cuda.is_available() else 'cpu',
                              weights_only=True)
        
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    sys.exit(1)

# ========== 创建模型 ==========
print("🔨 创建模型架构...")
try:
    # 导入ConvNeXt模型
    from models.convnext import convnext_tiny
    
    # 创建模型
    model = convnext_tiny(num_classes=5)
    
    # 加载权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
        print("✅ 使用CUDA")
    else:
        print("⚠️ 使用CPU")
        
except Exception as e:
    print(f"❌ 模型创建失败: {e}")
    sys.exit(1)

# ========== 准备测试数据 ==========
print(f"\n📁 加载测试数据: {TEST_DIR}")

# 测试时使用的transform（和训练时一样）
test_transform = transforms.Compose([
    transforms.Resize(235),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 创建数据集
dataset = ImageFolder(TEST_DIR, transform=test_transform)
print(f"📊 测试集大小: {len(dataset)} 张图片")
print(f"🎯 类别数量: {len(dataset.classes)}")

# 创建数据加载器
dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

# ========== 开始测试 ==========
print("\n🚀 开始测试...")
print("=" * 60)

all_predictions = []
all_labels = []
correct = 0
total = 0

# 逐批次测试
with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(dataloader):
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        
        # 前向传播
        outputs = model(images)
        
        # 获取预测结果
        _, predicted = torch.max(outputs, 1)
        
        # 统计正确率
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        # 保存结果用于分析
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # 显示进度
        if (batch_idx + 1) % 10 == 0:
            print(f"  已处理: {total}/{len(dataset)} 张图片")

# ========== 计算指标 ==========
accuracy = 100.0 * correct / total

print("\n" + "=" * 60)
print("🎯 测试结果")
print("=" * 60)

print(f"📈 总体准确率: {correct}/{total} = {accuracy:.2f}%")
print(f"📊 测试样本数: {total}")

# ========== 各类别准确率 ==========
print(f"\n📋 各类别准确率:")
print("-" * 60)

# 计算每个类别的准确率
class_correct = [0] * len(CLASS_NAMES)
class_total = [0] * len(CLASS_NAMES)

for true_label, pred_label in zip(all_labels, all_predictions):
    class_total[true_label] += 1
    if true_label == pred_label:
        class_correct[true_label] += 1

# 打印每个类别的结果
for i, class_name in enumerate(CLASS_NAMES):
    if class_total[i] > 0:
        class_acc = 100.0 * class_correct[i] / class_total[i]
        print(f"  {class_name}: {class_correct[i]}/{class_total[i]} = {class_acc:.1f}%")
    else:
        print(f"  {class_name}: 0/0 = N/A")

# ========== 混淆矩阵（简化版） ==========
print(f"\n🔀 混淆矩阵（错误分析）:")
print("-" * 60)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(all_labels, all_predictions)

# 打印错误分类
for i in range(len(CLASS_NAMES)):
    for j in range(len(CLASS_NAMES)):
        if i != j and cm[i, j] > 0:
            print(f"  {CLASS_NAMES[i]} → {CLASS_NAMES[j]}: {cm[i, j]} 个")

# ========== 保存结果 ==========
print(f"\n💾 保存测试结果...")
import json
from datetime import datetime

results = {
    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'model_path': MODEL_PATH,
    'test_dir': TEST_DIR,
    'total_samples': total,
    'correct_predictions': correct,
    'accuracy': accuracy,
    'class_accuracy': {
        CLASS_NAMES[i]: f"{100.0 * class_correct[i] / class_total[i]:.1f}%" 
        if class_total[i] > 0 else "N/A"
        for i in range(len(CLASS_NAMES))
    },
    'confusion_matrix': cm.tolist()
}

with open('test_results.json', 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"✅ 结果已保存到: test_results.json")

# ========== 单张图片测试函数 ==========
def test_single_image(image_path):
    """测试单张图片"""
    print(f"\n🔍 测试单张图片: {image_path}")
    
    try:
        # 加载图片
        img = Image.open(image_path).convert('RGB')
        
        # 预处理
        img_tensor = test_transform(img).unsqueeze(0)
        
        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()
        
        # 推理
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)
        
        # 获取top-3结果
        probs, indices = torch.topk(probabilities, 3)
        
        print("📊 预测结果:")
        for i in range(3):
            class_idx = indices[0][i].item()
            prob = probs[0][i].item() * 100
            print(f"  {i+1}. {CLASS_NAMES[class_idx]}: {prob:.1f}%")
            
        return {
            'top_class': CLASS_NAMES[indices[0][0].item()],
            'top_probability': probs[0][0].item() * 100
        }
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return None

# ========== 可选：测试单张图片 ==========
if len(sys.argv) > 1:
    test_image = sys.argv[1]
    if os.path.exists(test_image):
        test_single_image(test_image)

print("\n✅ 测试完成！")