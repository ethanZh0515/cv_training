import os, json
from glob import glob
from PIL import Image

# 1. 配置路径
ROOT = 'datasets/bdd100k'
JSON_DIRS = {
    'train': os.path.join(ROOT, 'labels/100k', 'train'),
    'val':   os.path.join(ROOT, 'labels/100k', 'val'),
}
IMG_DIRS = {
    'train': os.path.join(ROOT, 'images', '100k', 'train'),
    'val':   os.path.join(ROOT, 'images', '100k', 'val'),
}
OUT_LABEL_DIR = {
    'train': os.path.join(ROOT, 'yolo_labels', 'train'),
    'val':   os.path.join(ROOT, 'yolo_labels', 'val'),
}

# 2. 类别名称到 ID 的映射（按需扩充）
category_map = {
    'car': 0,
    'bus': 1,
    'person': 2,
    'bike': 3,
    'truck': 4,
    'motor': 5,
    'train': 6,
    'rider': 7,
    'traffic sign': 8,
    'traffic light': 9
    # 你可以继续添加：'car':2, 'rider':3, ...
}

# 3. 确保输出目录存在
for split in OUT_LABEL_DIR:
    os.makedirs(OUT_LABEL_DIR[split], exist_ok=True)

# 4. 主循环
for split in ['train', 'val']:
    json_paths = glob(os.path.join(JSON_DIRS[split], '*.json'))
    for jpath in json_paths:
        data = json.load(open(jpath, 'r'))
        img_name = data['name'] + '.jpg'
        img_path = os.path.join(IMG_DIRS[split], img_name)
        if not os.path.isfile(img_path):
            # 跳过找不到图片的
            continue

        # 动态读取图像尺寸
        with Image.open(img_path) as im:
            W, H = im.size

        yolo_lines = []
        for obj in data['frames'][0]['objects']:
            cat = obj.get('category', '')
            # 忽略 area/* 和 lane/* 类别
            if cat.startswith('area/') or cat.startswith('lane/'): # class to ignore for detection task(they were meant for segmentation tasks)
                continue
            if 'box2d' not in obj or cat not in category_map:
                continue

            cls_id = category_map[cat]
            b = obj['box2d']
            x1, y1, x2, y2 = b['x1'], b['y1'], b['x2'], b['y2']
            # 计算相对中心与长宽
            xc = ((x1 + x2) / 2) / W
            yc = ((y1 + y2) / 2) / H
            w  = (x2 - x1) / W
            h  = (y2 - y1) / H
            yolo_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        # 写入 .txt（若无目标则生成空文件）
        out_path = os.path.join(OUT_LABEL_DIR[split], data['name'] + '.txt')
        with open(out_path, 'w') as fout:
            fout.write('\n'.join(yolo_lines))

    print(f"[{split}] 转换完成：{len(json_paths)} 个 JSON → {OUT_LABEL_DIR[split]}")