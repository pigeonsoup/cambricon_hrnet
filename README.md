# HRNet-Image-Classification
本项目为 High-resolution networks (HRNets) for Image classification 在 MLU 卡上的适配，原始版本可参考[原始链接](https://github.com/HRNet/HRNet-Image-Classification)。这里对 HRNet-W18-C-Small-v2 模型进行了适配，参数量 15.6M。
# 使用
1. 检查 _run.sh_ 脚本中的参数配置，如使用第几块 MLU 卡 `export MLU_VISIBLE_DEVICES=0`，权重文件可修改 `$weight_dir` 对应位置，图片大小和均值标准差信息也可在对应位置设置。
2. 运行推理 `./run.sh`。
3. 检查 _valid.sh_ 脚本中同样的参数配置。
4. 运行验证测试集 `./valid.sh`。
# 实验
## 运行参数
```
batch size: 4
core number: 4
quantized mode: INT8
tensor type: HalfTensor
model: HRNet-W18-C-Small-v2
#params: 15.6M
dataset: ImageNet Val
```
## 结果
### 精度
| Device | Loss | Error@1 | Error@5 | Accuracy@1 | Accuracy@5 |
| --: | :--: | :--: | :--: | :--: | :--: |
| CPU | 0.9847 | 24.874 | 7.584 | 75.126 | 92.416 |
| MLU | 1.0211 | 25.850 | 8.140 | 74.150 | 91.860 |
### MLU 性能
```
Throughput: 1055 FPS
```