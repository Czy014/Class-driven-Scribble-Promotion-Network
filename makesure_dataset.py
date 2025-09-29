# 每个目录中所包含的文件个数都不一样。。。train.txt需要修改

import os

# 配置
txt_path = "/home/zhangsanyi/caizhengyang/Class-driven-Scribble-Promotion-Network/dataset/ScribbleSup/VOC2012/ImageSets/SegmentationAug/val.txt"  # 输入的txt文件
dirs = [
    # "/home/zhangsanyi/caizhengyang/Class-driven-Scribble-Promotion-Network/dataset/ScribbleSup/VOC2012/bmp_pseudolabel",
    # "/home/zhangsanyi/caizhengyang/Class-driven-Scribble-Promotion-Network/dataset/ScribbleSup/VOC2012/distance_pseudo/distance_CAM_lambd_e6",
    "/home/zhangsanyi/caizhengyang/Class-driven-Scribble-Promotion-Network/dataset/ScribbleSup/VOC2012/JPEGImages",
    # "/home/zhangsanyi/caizhengyang/Class-driven-Scribble-Promotion-Network/dataset/ScribbleSup/VOC2012/pascal_2012_scribble",
    # "/home/zhangsanyi/caizhengyang/Class-driven-Scribble-Promotion-Network/dataset/ScribbleSup/VOC2012/distance_pseudo/distance_map_lambd_e",
    "/home/zhangsanyi/caizhengyang/Class-driven-Scribble-Promotion-Network/dataset/ScribbleSup/VOC2012/SegmentationClassAug",
]  # 需要检查的目录列表
exts = [".jpg", ".png"]  # 可能的扩展名


def has_file(prefix):
    for d in dirs:
        found = False
        for ext in exts:
            if os.path.exists(os.path.join(d, prefix + ext)):
                found = True
                break
        if not found:
            return False
    return True


with open(txt_path, "r") as f:
    prefixes = [line.strip() for line in f if line.strip()]

valid_prefixes = [p for p in prefixes if has_file(p)]

# 覆盖原txt
with open(txt_path, "w") as f:
    for p in valid_prefixes:
        f.write(p + "\n")

print(f"筛选后剩余 {len(valid_prefixes)} 个前缀。")
