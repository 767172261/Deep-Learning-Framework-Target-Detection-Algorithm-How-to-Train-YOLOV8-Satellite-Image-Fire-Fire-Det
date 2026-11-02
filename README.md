![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3642d2cb9c1d44e681a2efc61171cb1c.png)
卫星图像火点火灾检测数据集、卫星火点烟雾检测数据集

| 数据集属性      | 详情                 |
|-----------------|----------------------|
| 总图像数量      | 30,480               |
| 标注格式        | YOLO, VOC            |
| 类别数量        | 2                    |
| 具体类别        | Smoke, Wildfire      |
| Smoke标注数量   | 1,311                |
| Wildfire标注数量 | 22,138               |

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c903d5970734421594d9c93c8de8f89f.png)


1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/47ae46107aff4d148585050661d85df4.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0bf0bdd3ec7a45cc885822744181b91f.png)




---

### ✅ 正确建议：使用 **YOLOv8** 进行训练

鉴于您的数据集已提供 **YOLO 格式标注**（即每张图像对应一个 `.txt` 文件，包含归一化的 `class_id center_x center_y width height`），推荐使用 **Ultralytics YOLOv8** 进行训练，这是目前最成熟、文档最完善的 YOLO 实现。

---

## 📌 使用 YOLOv8 训练卫星火点/烟雾检测模型的完整代码

### 1. 安装依赖
```bash
pip install ultralytics
```

### 2. 准备数据集目录结构
```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/ (可选)
├── labels/
│   ├── train/
│   ├── val/
│   └── test/ (可选)
└── fire_smoke.yaml
```

### 3. 创建 `fire_smoke.yaml` 配置文件
```yaml
# fire_smoke.yaml
path: ./dataset          # 数据集根目录
train: images/train      # 训练集图像路径（相对于 path）
val: images/val          # 验证集图像路径

# 类别数量
nc: 2

# 类别名称（顺序必须与标注中的 class_id 对应）
names: ['Smoke', 'Wildfire']
```

> ⚠️ 注意：YOLO 标注中 `Smoke` 应为 `class_id = 0`，`Wildfire` 为 `class_id = 1`，确保与 `names` 顺序一致。

### 4. Python 训练代码（YOLOv8）

```python
from ultralytics import YOLO

# 加载预训练模型（例如 YOLOv8n，也可选 s/m/l/x）
model = YOLO('yolov8n.pt')  # 或 yolov8s.pt, yolov8m.pt 等

# 开始训练
results = model.train(
    data='fire_smoke.yaml',     # 数据配置文件
    epochs=100,                 # 训练轮数
    imgsz=640,                  # 输入图像尺寸
    batch=16,                   # 批大小（根据显存调整）
    name='fire_smoke_yolov8n',  # 实验名称
    device=0,                   # 使用 GPU 0（可设为 'cpu' 或 [0,1] 多卡）
    workers=8,                  # 数据加载线程数
    patience=15,                # 早停耐心值（若验证指标不再提升）
    save=True,                  # 保存最佳和最后模型
    exist_ok=True               # 允许覆盖同名实验目录
)

# 可选：验证模型
metrics = model.val()
print(metrics.box.map)  # 输出 mAP50-95
```

### 5. 推理示例
```python
# 加载训练好的模型
best_model = YOLO('runs/detect/fire_smoke_yolov8n/weights/best.pt')

# 对单张图像推理
results = best_model('test_image.jpg')
results[0].show()  # 显示带框结果
```

---


