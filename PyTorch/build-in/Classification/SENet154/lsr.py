
import torch
import torch.nn as nn


class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, num_classes=1000, smooth_factor=0.0):
        """
        Constructor for the LabelSmoothing module.

        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.confidence = 1.0 - smooth_factor
        self.smoothing = smooth_factor

    def forward(self, x, target):
        target = target.to(torch.int64)
        
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1).to(torch.int64))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


if __name__ == '__main__':
    # 测试CPU/GPU版本
    x = torch.randn(2, 10)
    x.requires_grad = True
    y = torch.randint(0, 10, size=(2,))
    
    # CPU测试
    m = LabelSmoothingCrossEntropy(10)
    l = m(x, y)
    l.backward()
    print('CPU test ce ok, loss is ', l)
    
    m = LabelSmoothingCrossEntropy(10, 0.1)
    l = m(x, y)
    l.backward()
    print('CPU test lsce ok, loss is ', l)
    
    # GPU测试（如果可用）
    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()
        m = LabelSmoothingCrossEntropy(10).cuda()
        l = m(x, y)
        l.backward()
        print('GPU test ce ok, loss is ', l)
        
        m = LabelSmoothingCrossEntropy(10, 0.1).cuda()
        l = m(x, y)
        l.backward()
        print('GPU test lsce ok, loss is ', l)    