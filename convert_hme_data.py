import pickle
import shutil
from pathlib import Path
import numpy as np

# 数据路径
p = Path("./datasets/aligned_mosi.pkl")

# 备份原始数据
backup = Path("./datasets/aligned_mosi_original_dict.pkl")

if not backup.exists():
    shutil.copyfile(p, backup)
    print("已备份原始数据到:", backup)

# 读取数据
data = pickle.load(open(p, "rb"))

print("原始数据最外层 keys:", data.keys())

# valid / dev 名字统一
if "dev" not in data and "valid" in data:
    data["dev"] = data["valid"]
    print("已添加 dev = valid")

def pick(split, names):
    """
    从 split 里面按候选名字找字段。
    比如 text 可能叫 raw_text，也可能叫 text。
    """
    for name in names:
        if name in split:
            return split[name], name
    raise KeyError(f"找不到字段，候选字段是: {names}，当前字段有: {list(split.keys())}")

def to_scalar_label(y):
    """
    把 label 转成单个数字。
    """
    arr = np.array(y).reshape(-1)
    return float(arr[0])

def convert_split(split_name, split):
    """
    把原始字典格式转成 HME 代码需要的格式：
    ((words, visual, acoustic), label_id, segment)
    """

    # 如果已经是 list/tuple 格式，就不重复转
    if isinstance(split, (list, tuple)):
        print(split_name, "已经是 list/tuple 格式，跳过转换")
        return split

    print(f"\n正在转换 {split_name}")
    print(f"{split_name} 原始字段:", split.keys())

    texts, text_key = pick(split, ["raw_text", "text", "words"])
    visuals, visual_key = pick(split, ["vision", "visual"])
    audios, audio_key = pick(split, ["audio", "acoustic"])
    labels, label_key = pick(split, ["regression_labels", "labels", "label"])

    if "id" in split:
        ids = split["id"]
    elif "ids" in split:
        ids = split["ids"]
    else:
        ids = list(range(len(labels)))

    n = len(labels)

    print(f"{split_name}:")
    print("  text字段:", text_key)
    print("  visual字段:", visual_key)
    print("  audio字段:", audio_key)
    print("  label字段:", label_key)
    print("  样本数:", n)

    new_split = []

    for i in range(n):
        words = texts[i]
        visual = visuals[i]
        acoustic = audios[i]
        label_id = to_scalar_label(labels[i])
        segment = ids[i]

        new_example = ((words, visual, acoustic), label_id, segment)
        new_split.append(new_example)

    return new_split

# 开始转换
new_data = {}

for split_name in ["train", "valid", "dev", "test"]:
    if split_name in data:
        new_data[split_name] = convert_split(split_name, data[split_name])

# 保存回 aligned_mosi.pkl
pickle.dump(new_data, open(p, "wb"))

print("\n转换完成！")
print("转换后 keys:", new_data.keys())
print("train 第一条类型:", type(new_data["train"][0]))
print("train 第一条长度:", len(new_data["train"][0]))
print("train 第一条结构:")
print(type(new_data["train"][0][0]), type(new_data["train"][0][1]), type(new_data["train"][0][2]))