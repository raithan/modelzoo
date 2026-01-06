# Adapted to tecorigin hardware
from examples import *

if __name__ == '__main__':
    cf = Config()
    cf.add_argument('--game', type=str, default='BreakoutNoFrameskip-v4')
    cf.add_argument('--use_device', type=str, default='use_npu')
    cf.add_argument('--device_id', type=int, default=0)
    cf.add_argument('--max_steps', type=int, default=2e7)
    cf.add_argument('--save_interval', type=int, default=0)
    cf.add_argument('--eval_interval', type=int, default=0)
    cf.add_argument('--log_interval', type=int, default=0)
    cf.add_argument('--tag', type=str, default=None)
    cf.add_argument('--pth_path', type=str, default='null')
    cf.add_argument('--status_path', type=str, default='null')
    cf.merge()

    param = dict(game=cf.game, max_steps=cf.max_steps, save_interval=cf.save_interval, eval_interval=cf.eval_interval,
                 log_interval=cf.log_interval, pth_path=cf.pth_path, status_path=cf.status_path, tag=cf.tag, device_id=cf.device_id, maxremark=categorical_dqn_pixel.__name__)

    mkdir('data')
    random_seed()
    select_device(cf.use_device, cf.device_id)
    categorical_dqn_pixel(**param)
    exit()
