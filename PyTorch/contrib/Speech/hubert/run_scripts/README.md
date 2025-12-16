## **模型与训练参数**
| 参数名              | 说明                                                     | 示例                                  |
| ---------------- | ------------------------------------------------------ | ----------------------------------- |
| `dataset_dir`    | **必选参数**。训练所需的数据集根目录路径。程序会在该目录下读取 wav 或特征文件。           | `/data/teco-data/hubert/LibriSpeech`                |
| `checkpoint_dir` | **必选参数**。模型 checkpoint 的保存目录。训练中模型会定期写入此处。             | `./checkpoints/hubert`              |
| `--resume`       | 指定从某个 checkpoint 文件继续训练（断点续训）。                         | `--resume checkpoints/ckpt_100k.pt` |
| `--warmstart`    | 是否从 **fairseq 官方 HuBERT 预训练模型** 初始化参数（相当于 warm start）。 | `--warmstart`                       |
| `--mask`         | 是否启用输入 masking（HuBERT 的 Masked Prediction 机制）。         | `--mask`                            |
| `--alpha`        | Mask 区域的 loss 权重（控制 masked loss 与 unmasked loss 比例）。   | `--alpha 1.0`                       |
