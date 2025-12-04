WORK_DIR=$(dirname $(realpath $0))/..
cd $WORK_DIR

# 训练参数配置
EPOCHS=1
BATCH_SIZE=1
SAM_WEIGHTS=weights/sam_vit_b_01ec64.pth
MODEL_TYPE=vit_b
DATA_DIR=data_example/VOCdevkit
SAVE_DIR=runs
DEVICE=0

# 执行训练
python run_scripts/run_sam.py \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --sam-weights $SAM_WEIGHTS \
    --model-type $MODEL_TYPE \
    --data $DATA_DIR \
    --save_dir $SAVE_DIR \
    --device $DEVICE \
    --point-prompt True \
    --box-prompt True

echo "训练完成!"    