from torch_sdaa.utils import cuda_migrate
import numpy as np
import torch
import cv2
import os
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import time
from tcap_dllogger import Logger, StdOutBackend, Verbosity
# 新增sdaa amp导入
from torch.sdaa import amp

json_logger = Logger([StdOutBackend(Verbosity.DEFAULT)])
json_logger.metadata("train.loss", {"unit": "", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
json_logger.metadata("train.ips", {"unit": "imgs/s", "format": ":.3f", "GOAL": "MAXIMIZE", "STAGE": "TRAIN"})

# Read data
data_dir=r"LabPicsV1//"  # Path to dataset (LabPics 1)
data=[]  # list of files in dataset
for ff, name in enumerate(os.listdir(data_dir+"Simple/Train/Image/")):  # go over all folder annotation
    data.append({"image":data_dir+"Simple/Train/Image/"+name,"annotation":data_dir+"Simple/Train/Instance/"+name[:-4]+".png"})


def read_batch(data):  # read random image and its annotaion from  the dataset (LabPics)
    #  select image
    ent  = data[np.random.randint(len(data))]  # choose random entry
    Img = cv2.imread(ent["image"])[...,::-1]  # read image
    ann_map = cv2.imread(ent["annotation"])  # read annotation

    # resize image
    r = np.min([1024 / Img.shape[1], 1024 / Img.shape[0]])  # scalling factor
    Img = cv2.resize(Img, (int(Img.shape[1] * r), int(Img.shape[0] * r)))
    ann_map = cv2.resize(ann_map, (int(ann_map.shape[1] * r), int(ann_map.shape[0] * r)),interpolation=cv2.INTER_NEAREST)

    # merge vessels and materials annotations
    mat_map = ann_map[:,:,0]  # material annotation map
    ves_map = ann_map[:,:,2]  # vessel  annotaion map
    mat_map[mat_map==0] = ves_map[mat_map==0]*(mat_map.max()+1)  # merge maps

    # Get binary masks and points
    inds = np.unique(mat_map)[1:]  # load all indices
    points= []
    masks = []
    for ind in inds:
        mask=(mat_map == ind).astype(np.uint8)  # make binary mask corresponding to index ind
        masks.append(mask)
        coords = np.argwhere(mask > 0)  # get all coordinates in mask
        yx = np.array(coords[np.random.randint(len(coords))])  # choose random point/coordinate
        points.append([[yx[1], yx[0]]])
    return Img,np.array(masks),np.array(points), np.ones([len(masks),1])

# Load model
sam2_checkpoint = "/data/bigc-data/ltb/sam2/checkpoints/sam2.1_hiera_tiny.pt"  # path to model weight
model_cfg = "/configs/sam2.1/sam2.1_hiera_t.yaml"  #  model config
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")  # load model
predictor = SAM2ImagePredictor(sam2_model)

# Set training parameters
predictor.model.sam_mask_decoder.train(True)  # enable training of mask decoder
predictor.model.sam_prompt_encoder.train(True)  # enable training of prompt encoder
'''
#The main part of the net is the image encoder, if you have good GPU you can enable training of this part by using:
predictor.model.image_encoder.train(True)
#Note that for this case, you will also need to scan the SAM2 code for “no_grad” commands and remove them (“ no_grad” blocks the gradient collection, which saves memory but prevents training).
'''
optimizer=torch.optim.AdamW(params=predictor.model.parameters(),lr=1e-5,weight_decay=4e-5)
# 替换为sdaa的amp GradScaler
scaler = amp.GradScaler()  # 原代码：torch.cuda.amp.GradScaler()

# Training loop
for epoch in range(1):
    batch_size=1
    start_time = time.time()
    for step in range(102):
        # 替换为sdaa的amp.autocast
        with amp.autocast():  # 原代码：torch.cuda.amp.autocast()
            image,mask,input_point, input_label = read_batch(data)  # load data batch
            if mask.shape[0]==0: continue  # ignore empty batches
            predictor.set_image(image)  # apply SAM image encoder to the image

            # prompt encoding
            mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(input_point, input_label, box=None, mask_logits=None, normalize_coords=True)
            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(points=(unnorm_coords, labels),boxes=None,masks=None,)

            # mask decoder
            batched_mode = unnorm_coords.shape[0] > 1  # multi object prediction
            high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
            low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),sparse_prompt_embeddings=sparse_embeddings,dense_prompt_embeddings=dense_embeddings,multimask_output=True,repeat_image=batched_mode,high_res_features=high_res_features,)
            prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])# Upscale the masks to the original image resolution

            # Segmentaion Loss caclulation
            gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
            prd_mask = torch.sigmoid(prd_masks[:, 0])# Turn logit map to probability map
            seg_loss = (-gt_mask * torch.log(prd_mask + 0.00001) - (1 - gt_mask) * torch.log((1 - prd_mask) + 0.00001)).mean()  # cross entropy loss

            # Score loss calculation (intersection over union) IOU
            inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
            iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
            score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
            loss=seg_loss+score_loss*0.05  # mix losses

        # apply back propogation
        predictor.model.zero_grad()  # empty gradient
        scaler.scale(loss).backward()  # Backpropogate
        scaler.step(optimizer)
        scaler.update()  # Mix precision

        if step%1000==0: torch.save(predictor.model.state_dict(), "model.torch");print("save model")

        # Display results
        if step==0: mean_iou=0
        mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())
        
        # === 日志 ===
        batch_time = time.time() - start_time
        ips = batch_size / batch_time
        json_logger.log(
            step=(epoch, step),
            data={
                "rank": os.environ.get("LOCAL_RANK", "0"),
                "train.loss": loss.item(),
                "train.ips": ips,
            },
            verbosity=Verbosity.DEFAULT,
        )