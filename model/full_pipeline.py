#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import shutil
import random
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
import sys
import gammatone.filters as gt
from scipy import ndimage

warnings.filterwarnings('ignore')


AUDIO_ROOT_DIR = "ShipsEar"          # 原始音频文件夹
CSV_FILE_PATH = "shipsEarMeta_90.csv"  # 元数据CSV

# ---------- 输出路径 ----------
FEATURE_OUTPUT_DIR = "audio_class_gammatone_images"               # 第一步：生成的特征图根目录
DATASET_OUTPUT_DIR = "ships_gammatone_dataset_split"              # 第二步：划分后的数据集根目录
SMALL_SAMPLE_LIST_FILE = "small_sample_file_list.txt"             # 第三步：小样本文件名清单

# ---------- 类别映射 ----------
CLASS_MAPPING = {
    "Ocean liner": "A_Large_Merchant_Ship",
    "RORO": "A_Large_Merchant_Ship",
    "Dredger": "A_Large_Merchant_Ship",
    "Motorboat": "B_Small_Speedboat",
    "Tugboat": "B_Small_Speedboat",
    "Pilot ship": "B_Small_Speedboat",
    "Sailboat": "B_Small_Speedboat",
    "Yacht": "B_Small_Speedboat",
    "High speed motorboat": "B_Small_Speedboat",
    "Zodiac": "B_Small_Speedboat",
    "Fishboat": "C_Fishing_Boat",
    "Trawler": "C_Fishing_Boat",
    "Mussel boat": "C_Fishing_Boat",
    "fishboat": "C_Fishing_Boat",
    "Passengers": "D_Ferry",
    "Natural ambient noise": "E_Ambient_Noise"
}

# ---------- 音频参数 ----------
TARGET_SR = 16000
WINDOW_DURATION = 0.5      # 窗口长度（秒）
OVERLAP_DURATION = 0.3     # 窗口重叠（秒）

# ---------- 伽马通特征参数 ----------
N_FILTERS = 128
FMIN = 20
FMAX = 8000
FRAME_LENGTH = 512
HOP_LENGTH = 256

# ---------- 图像参数 ----------
DPI = 150
IMAGE_SIZE = (224, 224)
FIG_SIZE = (IMAGE_SIZE[0]/DPI, IMAGE_SIZE[1]/DPI)

# ---------- 划分比例 ----------
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# TEST_RATIO = 0.15 自动计算
RANDOM_SEED = 42

# ============================================================
# 2. 工具函数（伽马通滤波器兼容版本）
# ============================================================

def get_erb_filters_compatible(sr, n_filters, f_min, f_max):
    """兼容不同版本 gammatone 库的滤波器生成"""
    try:
        center_freqs = gt.erb_space(f_min, f_max, n_filters)
        erb_filters = gt.make_erb_filters(sr, center_freqs)
        return erb_filters, center_freqs
    except TypeError:
        try:
            erb_filters = gt.make_erb_filters(sr, n_filters, f_min, f_max)
            center_freqs = gt.erb_space(f_min, f_max, n_filters)
            return erb_filters, center_freqs
        except:
            erb_filters = gt.make_erb_filters(sr, n_filters)
            center_freqs = gt.erb_space(f_min, f_max, n_filters)
            return erb_filters, center_freqs

# ============================================================
# 3. 伽马通特征提取器
# ============================================================

class GammatoneFeatureExtractor:
    def __init__(self, sr=16000):
        self.sr = sr
        self.n_filters = N_FILTERS
        self.fmin = FMIN
        self.fmax = FMAX
        self.frame_length = FRAME_LENGTH
        self.hop_length = HOP_LENGTH
        self.erb_filters, self.center_freqs = get_erb_filters_compatible(
            sr, n_filters=N_FILTERS, f_min=FMIN, f_max=FMAX
        )

    def compute_gammatone_spec(self, audio):
        try:
            filtered_audio = gt.erb_filterbank(audio, self.erb_filters)
            audio_energy = np.square(filtered_audio)
            if audio_energy.ndim == 2:
                framed_energy = librosa.util.frame(
                    audio_energy,
                    frame_length=self.frame_length,
                    hop_length=self.hop_length,
                    axis=1
                )
                if framed_energy.ndim == 3:
                    smoothed_energy = framed_energy.mean(axis=2)
                else:
                    smoothed_energy = framed_energy.mean(axis=1)
            else:
                framed_energy = librosa.util.frame(
                    audio_energy,
                    frame_length=self.frame_length,
                    hop_length=self.hop_length
                )
                smoothed_energy = framed_energy.mean(axis=1)
            smoothed_energy = np.clip(smoothed_energy, 1e-10, np.inf)
            gammatone_db = librosa.power_to_db(smoothed_energy, ref=np.max)
            gammatone_norm = (gammatone_db - gammatone_db.min()) / (gammatone_db.max() - gammatone_db.min() + 1e-8)
            gammatone_enhanced = np.power(gammatone_norm, 0.7) * 255
            gammatone_enhanced = gammatone_enhanced.astype(np.uint8)
            if gammatone_enhanced.ndim == 2:
                gammatone_transposed = gammatone_enhanced.T
            else:
                gammatone_transposed = gammatone_enhanced.reshape(-1, gammatone_enhanced.shape[-1]).T
            gammatone_resized = self.resize_feature(gammatone_transposed, IMAGE_SIZE)
            return gammatone_resized
        except Exception as e:
            print(f"❌ 伽马通特征计算失败: {e}")
            return None

    def resize_feature(self, feat, target_size):
        if feat.ndim == 1:
            feat = feat.reshape(1, -1)
        h, w = feat.shape
        target_h, target_w = target_size
        scale = min(target_h / h, target_w / w)
        new_h, new_w = int(h * scale), int(w * scale)
        if new_h == h and new_w == w:
            resized_feat = feat
        else:
            resized_feat = ndimage.zoom(feat, (scale, scale), order=1)
        padded_feat = np.zeros(target_size, dtype=np.uint8)
        pad_h = (target_h - new_h) // 2
        pad_w = (target_w - new_w) // 2
        pad_h = max(0, pad_h)
        pad_w = max(0, pad_w)
        new_h = min(new_h, target_h - pad_h)
        new_w = min(new_w, target_w - pad_w)
        padded_feat[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized_feat[:new_h, :new_w]
        return padded_feat

# ============================================================
# 4. 音频处理器
# ============================================================

class AudioProcessor:
    def __init__(self, target_sr=16000, window_duration=0.5, overlap_duration=0.3):
        self.target_sr = target_sr
        self.window_duration = window_duration
        self.overlap_duration = overlap_duration
        self.extractor = GammatoneFeatureExtractor(sr=target_sr)

    def process_audio(self, audio_path, output_dir, audio_basename, max_segments=None):
        try:
            y, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)
            y = y.astype(np.float32)
            y = librosa.effects.trim(y, top_db=20)[0]
            if len(y) < self.window_duration * sr:
                return 0
            window_samples = int(self.window_duration * sr)
            overlap_samples = int(self.overlap_duration * sr)
            step_samples = window_samples - overlap_samples
            if step_samples <= 0:
                step_samples = window_samples // 2
            total_samples = len(y)
            start_sample = 0
            processed_segments = 0
            segment_idx = 0
            while start_sample + window_samples <= total_samples:
                if max_segments and processed_segments >= max_segments:
                    break
                y_segment = y[start_sample:start_sample + window_samples]
                gammatone_feat = self.extractor.compute_gammatone_spec(y_segment)
                if gammatone_feat is None:
                    start_sample += step_samples
                    segment_idx += 1
                    continue
                save_filename = f"{audio_basename}_gammatone_seg{segment_idx+1:03d}.png"
                save_path = os.path.join(output_dir, save_filename)
                success = self.plot_single_feature(gammatone_feat, save_path)
                if success:
                    processed_segments += 1
                segment_idx += 1
                start_sample += step_samples
            return processed_segments
        except Exception as e:
            print(f"  ✗ 音频处理失败: {e}")
            return 0

    def plot_single_feature(self, feat, save_path):
        if feat is None:
            return False
        fig = plt.figure(figsize=FIG_SIZE, dpi=DPI, frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(feat, cmap='turbo', aspect='auto', interpolation='nearest')
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight', pad_inches=0, facecolor='black')
        plt.close(fig)
        return True

# ============================================================
# 5. 功能一：生成 Gammatone 特征图
# ============================================================

def step1_generate_gammatone_images():
    print("\n" + "="*70)
    print("第 1 步：从原始音频生成 Gammatone 特征图")
    print("="*70)
    
    # 清空并创建输出目录
    if os.path.exists(FEATURE_OUTPUT_DIR):
        shutil.rmtree(FEATURE_OUTPUT_DIR)
    os.makedirs(FEATURE_OUTPUT_DIR, exist_ok=True)
    
    unique_classes = list(set(CLASS_MAPPING.values()))
    for cls in unique_classes:
        cls_dir = os.path.join(FEATURE_OUTPUT_DIR, cls)
        os.makedirs(cls_dir, exist_ok=True)
    print(f"✅ 创建 {len(unique_classes)} 个类别文件夹")
    
    # 加载CSV
    try:
        df = pd.read_csv(CSV_FILE_PATH)
        if 'filename' not in df.columns or 'type' not in df.columns:
            raise ValueError("CSV 缺少 filename 或 type 列")
        df['mapped_class'] = df['type'].map(CLASS_MAPPING)
        df = df.dropna(subset=['mapped_class'])
        print(f"✅ 加载CSV成功，有效音频记录数: {len(df)}")
    except Exception as e:
        print(f"❌ 加载CSV失败: {e}")
        return False
    
    processor = AudioProcessor(TARGET_SR, WINDOW_DURATION, OVERLAP_DURATION)
    class_stats = {cls: {'files': 0, 'segments': 0} for cls in unique_classes}
    total_processed_files = 0
    total_generated_images = 0
    
    for target_class in tqdm(unique_classes, desc="处理类别"):
        class_df = df[df['mapped_class'] == target_class]
        if len(class_df) == 0:
            continue
        output_dir = os.path.join(FEATURE_OUTPUT_DIR, target_class)
        for idx, row in enumerate(class_df.itertuples(), 1):
            audio_filename = row.filename
            audio_path = os.path.join(AUDIO_ROOT_DIR, audio_filename)
            if not os.path.exists(audio_path):
                print(f"  ⚠ 文件不存在: {audio_filename}")
                continue
            audio_basename = os.path.splitext(audio_filename)[0]
            segments = processor.process_audio(audio_path, output_dir, audio_basename)
            if segments > 0:
                total_processed_files += 1
                total_generated_images += segments
                class_stats[target_class]['files'] += 1
                class_stats[target_class]['segments'] += segments
    
    print(f"\n✅ 特征图生成完成！共处理 {total_processed_files} 个音频，生成 {total_generated_images} 张图片")
    print(f"📁 输出目录: {os.path.abspath(FEATURE_OUTPUT_DIR)}")
    return True

# ============================================================
# 6. 功能二：划分数据集 (7:1.5:1.5)
# ============================================================

def step2_split_dataset():
    print("\n" + "="*70)
    print("第 2 步：按 7:1.5:1.5 划分训练/验证/测试集")
    print("="*70)
    
    SOURCE_DIR = FEATURE_OUTPUT_DIR
    OUTPUT_DIR = DATASET_OUTPUT_DIR
    random.seed(RANDOM_SEED)
    
    # 创建输出目录
    train_dir = os.path.join(OUTPUT_DIR, "train")
    val_dir = os.path.join(OUTPUT_DIR, "val")
    test_dir = os.path.join(OUTPUT_DIR, "test")
    for d in [train_dir, val_dir, test_dir]:
        os.makedirs(d, exist_ok=True)
    
    # 统计原始数据
    class_stats = {}
    for cls in os.listdir(SOURCE_DIR):
        cls_path = os.path.join(SOURCE_DIR, cls)
        if os.path.isdir(cls_path):
            images = [f for f in os.listdir(cls_path) if f.endswith('.png')]
            class_stats[cls] = len(images)
    print("原始各类别图片数:", class_stats)
    
    total_train = total_val = total_test = 0
    for cls, count in class_stats.items():
        src_path = os.path.join(SOURCE_DIR, cls)
        images = [f for f in os.listdir(src_path) if f.endswith('.png')]
        random.shuffle(images)
        
        # 确定比例（对 Fishing_Boat 特殊处理）
        if cls == "C_Fishing_Boat":
            train_ratio = 0.65
            val_ratio = 0.175
        else:
            train_ratio = TRAIN_RATIO
            val_ratio = VAL_RATIO
        test_ratio = 1.0 - train_ratio - val_ratio
        
        train_end = int(len(images) * train_ratio)
        val_end = train_end + int(len(images) * val_ratio)
        
        train_imgs = images[:train_end]
        val_imgs = images[train_end:val_end]
        test_imgs = images[val_end:]
        
        # 复制到目标目录
        for split, img_list in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
            dst_dir = os.path.join(OUTPUT_DIR, split, cls)
            os.makedirs(dst_dir, exist_ok=True)
            for img in img_list:
                shutil.copy2(os.path.join(src_path, img), os.path.join(dst_dir, img))
        
        total_train += len(train_imgs)
        total_val += len(val_imgs)
        total_test += len(test_imgs)
        print(f"{cls}: 训练 {len(train_imgs)} | 验证 {len(val_imgs)} | 测试 {len(test_imgs)}")
    
    print(f"\n✅ 数据集划分完成！总样本: {total_train+total_val+total_test}")
    print(f"   训练: {total_train} | 验证: {total_val} | 测试: {total_test}")
    print(f"📁 输出目录: {os.path.abspath(OUTPUT_DIR)}")
    return True

# ============================================================
# 7. 功能三：提取小样本文件名（从训练集中取 7%）
# ============================================================

def step3_extract_small_sample_list():
    print("\n" + "="*70)
    print("第 3 步：提取小样本训练集（7%）的文件名清单")
    print("="*70)
    
    # 使用上一步划分好的训练集
    train_root = os.path.join(DATASET_OUTPUT_DIR, "train")
    small_sample_files = set()
    
    # 从训练集每个类别中取 7%（按文件数）
    for cls in os.listdir(train_root):
        cls_path = os.path.join(train_root, cls)
        if not os.path.isdir(cls_path):
            continue
        all_files = [f for f in os.listdir(cls_path) if f.endswith('.png')]
        # 按文件名排序保证可复现
        all_files.sort()
        n_small = max(1, int(len(all_files) * 0.07))   # 至少取1张
        selected = all_files[:n_small]
        small_sample_files.update(selected)
        print(f"{cls}: 共 {len(all_files)} 张，抽取 {len(selected)} 张作为小样本")
    
    # 保存清单
    with open(SMALL_SAMPLE_LIST_FILE, "w") as f:
        for img in sorted(small_sample_files):
            f.write(f"{img}\n")
    print(f"✅ 小样本文件名清单已保存到: {SMALL_SAMPLE_LIST_FILE}")
    print(f"   总计 {len(small_sample_files)} 张图片")
    return True

# ============================================================
# 8. 主流程
# ============================================================

def main():
    print("\n" + "="*70)
    print("水下声学信号分类全流程 (Gammatone 特征提取 + 数据集划分 + 小样本提取)")
    print("="*70)
    print("当前配置:")
    print(f"  - 音频路径: {AUDIO_ROOT_DIR}")
    print(f"  - CSV文件: {CSV_FILE_PATH}")
    print(f"  - 特征图输出: {FEATURE_OUTPUT_DIR}")
    print(f"  - 划分数据集输出: {DATASET_OUTPUT_DIR}")
    print(f"  - 小样本清单文件: {SMALL_SAMPLE_LIST_FILE}")
    print("="*70)
    
    # 检查依赖
    required_libs = ['librosa', 'pandas', 'matplotlib', 'numpy', 'gammatone', 'scipy']
    missing = []
    for lib in required_libs:
        try:
            __import__(lib)
        except ImportError:
            missing.append(lib)
    if missing:
        print(f"❌ 缺少依赖库: {', '.join(missing)}")
        print(f"请安装: pip install {' '.join(missing)}")
        sys.exit(1)
    
    # 按顺序执行
    success = step1_generate_gammatone_images()
    if not success:
        print("❌ 第1步失败，终止流程")
        sys.exit(1)
    
    success = step2_split_dataset()
    if not success:
        print("❌ 第2步失败，终止流程")
        sys.exit(1)
    
    success = step3_extract_small_sample_list()
    if not success:
        print("❌ 第3步失败")
    
    print("\n" + "="*70)
    print("🎉 全流程执行完毕！")
    print("="*70)
    print("生成内容:")
    print(f"  1. Gammatone 特征图: {os.path.abspath(FEATURE_OUTPUT_DIR)}")
    print(f"  2. 划分后的数据集: {os.path.abspath(DATASET_OUTPUT_DIR)} (包含 train/val/test)")
    print(f"  3. 小样本文件名清单: {os.path.abspath(SMALL_SAMPLE_LIST_FILE)}")
    print("="*70)

if __name__ == "__main__":
    main()