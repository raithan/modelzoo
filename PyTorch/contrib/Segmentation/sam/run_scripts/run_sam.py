from torch_sdaa.utils import cuda_migrate
from torch_sdaa import amp  # ✅ 启用 AMP
import os
import sys
import time
import torch
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path
import faulthandler
faulthandler.enable()

# 添加工作目录到系统路径（确保能找到utils包）
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
sys.path.append(str(ROOT))

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

# 导入参数解析
from argument import parse_opt

# 核心函数导入
from utils.general import get_random_prompts, mask2one_hot
from utils.custom_dataset import CustomDataset
from utils.loss import soft_dice_loss


def main(opt):
    log_cnt = 0
    MAX_LOG = 100

    # === 数据加载 ===
    data_folder = opt.data
    dataset = CustomDataset(data_folder, txt_name="trainval.txt")
    batch_size = opt.batch_size
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=CustomDataset.custom_collate
    )

    # === 基础配置 ===
    sam_checkpoint = opt.sam_weights
    model_type = opt.model_type
    device = "cuda"
    save_dir = opt.save_dir
    os.makedirs(save_dir, exist_ok=True)
    num_epochs = opt.epochs
    point_prompt = opt.point_prompt
    box_prompt = opt.box_prompt
    point_box = (point_prompt and box_prompt)

    # === 日志系统 ===
    from tcap_dllogger import Logger, StdOutBackend, Verbosity
    json_logger = Logger([StdOutBackend(Verbosity.DEFAULT)])
    json_logger.metadata("train.loss", {"unit": "", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
    json_logger.metadata("train.ips", {"unit": "imgs/s", "format": ":.3f", "GOAL": "MAXIMIZE", "STAGE": "TRAIN"})

    # === 模型初始化 ===
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam.train()
    predictor = SamPredictor(sam)
    print(f"Finished loading SAM model")

    # === 优化器与调度器 ===
    lr = 1e-6
    weight_decay = 1e-4
    optimizer = torch.optim.AdamW(
        sam.mask_decoder.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)

    # === 损失函数 ===
    BCEseg = torch.nn.BCEWithLogitsLoss().to(device)  # ✅ 改为安全版本
    losses = []
    best_loss = 1e10

    model_transform = ResizeLongestSide(sam.image_encoder.img_size)
    scaler = amp.GradScaler()  # ✅ AMP 梯度缩放器

    # === 训练循环 ===
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        optimizer.zero_grad()

        for idx, (images, gts, image_names) in enumerate(tqdm(dataloader)):
            if idx >= 101:
                break

            start_time = time.time()
            valid_classes = []

            for i in range(images.shape[0]):
                image = images[i]
                original_size = image.shape[:2]
                input_size = model_transform.get_preprocess_shape(
                    image.shape[0], image.shape[1], sam.image_encoder.img_size
                )
                gt = gts[i].copy()
                gt_classes = np.unique(gt)
                predictions = []

                # 提取特征
                with torch.no_grad():
                    predictor.set_image(image, "RGB")
                    image_embedding = predictor.get_image_embedding()

                # 为每个类别生成提示
                for cls in gt_classes:
                    if isinstance(cls, torch.Tensor):
                        cls = cls.item()
                    if cls == 0 or cls == 255:
                        continue
                    valid_classes.append(cls)
                    (foreground_points, background_points), bbox = get_random_prompts(gt, cls)
                    if len(foreground_points) == 0:
                        continue

                    # 点提示
                    if not point_prompt:
                        points = None
                    else:
                        all_points = np.concatenate((foreground_points, background_points), axis=0)
                        point_labels = np.array(
                            [1] * foreground_points.shape[0] + [0] * background_points.shape[0],
                            dtype=int
                        )
                        all_points = model_transform.apply_coords(all_points, original_size)
                        all_points = torch.as_tensor(all_points, dtype=torch.float, device=device)
                        point_labels = torch.as_tensor(point_labels, dtype=torch.float, device=device)
                        all_points, point_labels = all_points[None, :, :], point_labels[None, :]
                        points = (all_points, point_labels)

                    # 框提示
                    if not box_prompt:
                        box_torch = None
                    else:
                        box = model_transform.apply_boxes(bbox, original_size)
                        box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
                        box_torch = box_torch[None, :]

                    # 随机丢弃提示
                    if point_box and np.random.random() < 0.5:
                        if np.random.random() < 0.25:
                            points = None
                        elif np.random.random() > 0.75:
                            box_torch = None

                    # prompt 编码
                    with torch.no_grad():
                        sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                            points=points,
                            boxes=box_torch,
                            masks=None,
                        )

                    # 第一次预测
                    mask_predictions, scores = sam.mask_decoder(
                        image_embeddings=image_embedding.to(device),
                        image_pe=sam.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=True,
                    )

                    mask_input = mask_predictions[:, torch.argmax(scores), ...].unsqueeze(1)

                    # 第二次 refine
                    with torch.no_grad():
                        sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                            points=points,
                            boxes=box_torch,
                            masks=mask_input,
                        )

                    mask_predictions, scores = sam.mask_decoder(
                        image_embeddings=image_embedding.to(device),
                        image_pe=sam.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                    )

                    best_mask = sam.postprocess_masks(mask_predictions, input_size, original_size)
                    predictions.append(best_mask)

                predictions = torch.cat(predictions, dim=1)

            # === 计算 loss（AMP 模式） ===
            gts = torch.from_numpy(gts).unsqueeze(1)
            gts_onehot = mask2one_hot(gts, valid_classes).to(device)

            with amp.autocast():
                # ✅ BCEWithLogitsLoss 内部自带 sigmoid
                loss_bce = BCEseg(predictions, gts_onehot)
                loss_dice = soft_dice_loss(predictions, gts_onehot, smooth=1e-5, activation='sigmoid')
                loss = loss_bce + loss_dice

            # === 反向传播与优化 ===
            optimizer.zero_grad()
            scaler.scale(loss).backward()      # ✅ AMP安全反向传播
            scaler.step(optimizer)             # ✅ 自动unscale + 更新参数
            scaler.update()                    # ✅ 更新缩放因子

            epoch_loss += loss.item()

            # === 日志 ===
            batch_time = time.time() - start_time
            ips = batch_size / batch_time
            json_logger.log(
                step=(epoch, idx),
                data={
                    "rank": os.environ.get("LOCAL_RANK", "0"),
                    "train.loss": loss.item(),
                    "train.ips": ips,
                },
                verbosity=Verbosity.DEFAULT,
            )
            print(f"epoch: {epoch} idx:{idx} loss: {loss.item():.6f}")

        # === 每个epoch结束 ===
        epoch_loss /= (idx + 1)
        losses.append(epoch_loss)
        scheduler.step()
        print(f'EPOCH: {epoch+1}, Loss: {epoch_loss:.6f}')

        # 保存最优权重
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            mask_decoder_weights = sam.mask_decoder.state_dict()
            mask_decoder_weights = {f"mask_decoder.{k}": v for k, v in mask_decoder_weights.items()}
            save_path = os.path.join(save_dir, f'sam_decoder_fintune_{epoch+1}_amp_safe.pth')
            torch.save(mask_decoder_weights, save_path)
            print(f"Saved best weights to {save_path}, epoch: {epoch+1}")

    print("训练完成!")


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
