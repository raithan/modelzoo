from argument import parse_args
import os

def build_hyper_parameters(args):
    launcher = args.launcher
    config = args.config

    hyper_parameters = f"{config} --launcher {launcher}"

    if args.amp:
        hyper_parameters += " --amp"

    if args.cfg_options:
        cfg_str = " ".join([f"{k}={v}" for k, v in args.cfg_options.items()])
        hyper_parameters += f" --cfg-options {cfg_str}"

    return hyper_parameters


def build_command(args, hyper_parameters):
    cmd = (
        f"python -m torch.distributed.launch --nnodes={args.nnodes} "
        f"--node_rank={args.node_rank} "
        f"--master_addr={args.master_addr} "
        f"--nproc_per_node={args.nproc_per_node} "
        f"--master_port={args.master_port} "
        f"tools/train.py {hyper_parameters}"
    )
    print("cmd--->>>>>:\n{}\n".format(cmd))
    return cmd


def excute_command(cmd):
    import subprocess
    try:
        subprocess.check_call(cmd, shell=True)
    except subprocess.CalledProcessError as e:
        exit_code = e.returncode
        print("Command failed with exit code:", exit_code)
        exit(exit_code)


if __name__ == "__main__":
    args = parse_args()
    hyper_parameters = build_hyper_parameters(args)
    cmd = build_command(args, hyper_parameters)
    excute_command(cmd)
