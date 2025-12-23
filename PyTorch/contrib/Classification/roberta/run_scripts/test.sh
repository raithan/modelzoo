#!/bin/bash
script_path=$(dirname $(readlink -f "$0"))
echo "当前脚本路径: $script_path"

#安装依赖
cd .. 
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install numpy==1.24.3
pip install huggingface_hub
pip install parameterized
cd $script_path
git clone https://gitee.com/xiwei777/tcap_dllogger.git
cd tcap_dllogger
python setup.py install
cd ..
#执行训练

mkdir -p roberta_out && python run_roberta.py \
--train_file ../configs/train_sample.txt \
--do_train --do_eval \
--output_dir roberta_out \
--overwrite_output_dir \
--per_device_train_batch_size 2 \
--max_seq_length 32 \
--line_by_line 2>&1 | tee sdaa.log

# 生成loss对比图
python loss.py --sdaa-log sdaa.log --cuda-log cuda.log
