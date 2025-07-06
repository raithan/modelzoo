import torch 
import torch_sdaa
import timm, os
os.environ['TORCH_SHOW_CPP_STACKTRACES'] = '1'   # 打印 C++ 栈
device = torch.device('sdaa')
m = timm.create_model('convnext_base', pretrained=False).to(device)
x = torch.randn(1, 3, 224, 224, device=device)
m(x)
