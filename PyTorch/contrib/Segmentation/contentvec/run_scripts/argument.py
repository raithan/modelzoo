# BSD 3-Clause License
# Copyright (c) 2023, Tecorigin Co., Ltd.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of the copyright holder nor the names of its contributors
#   may be used to endorse or promote products derived from this software
#   without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import argparse

def parse_options():
    parser = argparse.ArgumentParser(description="Training arguments for ContentVec")

    # ================= Required =================
    parser.add_argument("--expdir", type=str, default="./exp",
                        help="Experiment directory for saving logs & checkpoints")
    parser.add_argument("--config_dir", type=str, default="./contentvec/config/contentvec",
                        help="Path to hydra config directory")
    parser.add_argument("--config_name", type=str, default="contentvec",
                        help="Config name for hydra")

    # ================= Data =================
    parser.add_argument("--task.data", type=str, default="/data/teco-data/LibriSpeech/metadata",
                        help="Path to dataset metadata")
    parser.add_argument("--task.label_dir", type=str, default="/data/teco-data/LibriSpeech/label",
                        help="Directory containing label files")
    parser.add_argument("--task.labels", type=str, default='["km"]',
                        help="List of labels to use")
    parser.add_argument("--task.spk2info", type=str, default="/data/teco-data/LibriSpeech/metadata/spk2info.dict",
                        help="Path to spk2info dictionary")
    parser.add_argument("--task.crop", type=str, default="true",
                        help="Whether to crop input")

    # ================= Dataset =================
    parser.add_argument("--dataset.train_subset", type=str, default="train")
    parser.add_argument("--dataset.num_workers", type=int, default=10)
    parser.add_argument("--dataset.max_tokens", type=int, default=500000)

    # ================= Training =================
    parser.add_argument("--optimization.update_freq", type=str, default="[1]")
    parser.add_argument("--optimization.max_update", type=int, default=200)
    parser.add_argument("--lr_scheduler.warmup_updates", type=int, default=8000)
    parser.add_argument("--common.log_interval", type=int, default=1)

    # ================= Model =================
    parser.add_argument("--model.label_rate", type=int, default=50)
    parser.add_argument("--model.encoder_layers_1", type=int, default=3)
    parser.add_argument("--model.logit_temp_ctr", type=float, default=0.1)
    parser.add_argument("--model.ctr_layers", type=str, default="[-6]")
    parser.add_argument("--model.extractor_mode", type=str, default="default")

    # ================= Checkpoint =================
    parser.add_argument("--checkpoint.keep_best_checkpoints", type=int, default=10)

    # ================= Criterion =================
    parser.add_argument("--criterion.loss_weights", type=str, default="[10,1e-5]")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_options()
    print(args)
