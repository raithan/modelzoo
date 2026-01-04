from __future__ import print_function, absolute_import
import time
import os
import torch
from torch.cuda import amp  # 添加AMP支持
from torch.autograd import Variable

from .evaluation_metrics import accuracy
from .utils.meters import AverageMeter
from .utils import Bar
from torch.nn import functional as F

if torch.cuda.is_available():
    print("CUDA is available. Training on GPU.")
    
    device = torch.device("cuda")
    
import tcap_dllogger
from tcap_dllogger import Logger, StdOutBackend, JSONStreamBackend, Verbosity
json_logger = Logger(
    [
        StdOutBackend(Verbosity.DEFAULT),
        JSONStreamBackend(Verbosity.VERBOSE, "dlloger_example.json"),
    ]
)
json_logger.metadata("train.loss", {"unit": "", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
json_logger.metadata("train.ips",{"unit": "imgs/s", "format": ":.3f", "GOAL": "MAXIMIZE", "STAGE": "TRAIN"})


class BaseTrainer(object):
    def __init__(self, model, criterion, X, Y, SMLoss_mode=0):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.scaler = amp.GradScaler(enabled=True)
        self.amp_enabled = True  # 控制是否启用AMP
        self.dynamic_scaling = True  # 启用动态缩放


    def train(self, epoch, data_loader, optimizer, print_freq=1 , step=None):
        self.model.train()


        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()
        end = time.time()

        bar = Bar('Processing', max=len(data_loader))
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, targets = self._parse_data(inputs)
            #with amp.autocast(enabled=self.amp_enabled):
            loss0, loss1, loss2, loss3, loss4, loss5, prec1 = self._forward(inputs, targets)
#=========================================================================
            loss = (loss0+loss1+loss2+loss3+loss4+loss5)/6

            losses.update(loss.data.item(), targets.size(0)) 
            #losses.update(loss.data[0], targets.size(0))
            precisions.update(prec1, targets.size(0))

            optimizer.zero_grad()

            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.scaler.step(optimizer)

            self.scaler.update()
            #loss.backward()
            # torch.autograd.backward([loss0, loss1, loss2, loss3, loss4, loss5],
            #             [torch.ones(()).to(device), torch.ones(()).to(device), 
            #              torch.ones(()).to(device), torch.ones(()).to(device),
            #              torch.ones(()).to(device), torch.ones(()).to(device)])
            
            #optimizer.step()
            torch.cuda.synchronize()
            batch_time.update(time.time() - end)
            end = time.time()
            if int(os.getenv('LOCAL_RANK', '0')) == 0:
            # plot progress
                bar.suffix = 'Epoch: [{N_epoch}][{N_batch}/{N_size}] | Time {N_bt:.3f} {N_bta:.3f} | Data {N_dt:.3f} {N_dta:.3f} | Loss {N_loss:.3f} {N_lossa:.3f} | Prec {N_prec:.2f} {N_preca:.2f}'.format(
                        N_epoch=epoch, N_batch=i + 1, N_size=len(data_loader),
                                N_bt=batch_time.val, N_bta=batch_time.avg,
                                N_dt=data_time.val, N_dta=data_time.avg,
                                N_loss=losses.val, N_lossa=losses.avg,
                                N_prec=precisions.val, N_preca=precisions.avg,
                                )
                bar.next()
                json_logger.log(
                step = (epoch, i),
                data = {
                    "rank":os.environ["LOCAL_RANK"],
                    "train.loss":losses.val
                    },
                verbosity=Verbosity.DEFAULT,)
            if step == i :
                print("runing step is {step}, end ")
                exit()
        
        if int(os.getenv('LOCAL_RANK', '0')) == 0:
            bar.finish()




    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError


class Trainer(BaseTrainer):
    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = [Variable(imgs.to(device))]
        targets = Variable(pids.to(device))
        return inputs, targets

    def _forward(self, inputs, targets):
        outputs = self.model(*inputs)

        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            loss0 = self.criterion(outputs[1][0], targets)
            loss1 = self.criterion(outputs[1][1], targets)
            loss2 = self.criterion(outputs[1][2], targets)
            loss3 = self.criterion(outputs[1][3], targets)
            loss4 = self.criterion(outputs[1][4], targets)
            loss5 = self.criterion(outputs[1][5], targets)
            
            # 确保所有输出都参与计算
            with torch.no_grad():
                prec, = accuracy(outputs[1][2].data, targets.data)
                prec = prec.item()

        # index = (targets-751).data.nonzero().squeeze_()
		
        # if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
        #     loss0 = self.criterion(outputs[1][0],targets)
        #     loss1 = self.criterion(outputs[1][1],targets)
        #     loss2 = self.criterion(outputs[1][2],targets)
        #     loss3 = self.criterion(outputs[1][3],targets)
        #     loss4 = self.criterion(outputs[1][4],targets)
        #     loss5 = self.criterion(outputs[1][5],targets)
        #     prec, = accuracy(outputs[1][2].data, targets.data)

        #     #prec = prec[0]
        #     prec = prec.item()
                        
        elif isinstance(self.criterion, OIMLoss):
            loss, outputs = self.criterion(outputs, targets)
            prec, = accuracy(outputs.data, targets.data)
            prec = prec[0]
        elif isinstance(self.criterion, TripletLoss):
            loss, prec = self.criterion(outputs, targets)
        else:
            raise ValueError("Unsupported loss:", self.criterion)
        return loss0, loss1, loss2, loss3, loss4, loss5, prec
