import json
import cv2
import os

# ==== 配置路径 ====
image_path = r"D:\zyx\pythonProject\cv_demos\cv-training\datasets\bdd100k\images\100k\train\0000f77c-62c2a288.jpg"
json_path = r"D:\zyx\pythonProject\cv_demos\cv-training\datasets\bdd100k\labels\100k\train\0000f77c-62c2a288.json"

# ==== 加载图像 ====
img = cv2.imread(image_path)

if img is None:
    raise FileNotFoundError(f"Image not found: {image_path}")
height, width, _ = img.shape

# ==== 加载 JSON ====
with open(json_path, 'r') as f:
    data = json.load(f)

# ==== 提取并绘制 box2d ====
for obj in data['frames'][0]['objects']:
    if 'box2d' not in obj:
        continue
    box = obj['box2d']
    category = obj.get('category', 'unknown')
    x1, y1, x2, y2 = int(box['x1']), int(box['y1']), int(box['x2']), int(box['y2'])

    # 绘制矩形框和类别文本
    cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
    cv2.putText(img, category, (x1, y1 - 5), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5, color=(0, 255, 0), thickness=1)

# ==== 显示或保存图像 ====
cv2.imshow("Box2D Visualization", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 如果你想保存图片而不是显示：
# cv2.imwrite("output.jpg", img)