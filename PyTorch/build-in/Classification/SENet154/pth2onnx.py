
import sys 
import torch
import torch.onnx


sys.path.append('.')

import senet 


device = 'cpu'


def convert(file_path):
    model = senet.senet154(num_classes=1000, pretrained='imagenet', use_pretrained=False).to(device)
    state_dict = torch.load(file_path, map_location=device)["net"]
    model.load_state_dict({k.replace('module.', '', 1): v for k, v in state_dict.items()})
    model.eval()

    input_names = ["actual_input_1"]
    output_names = ["output1"]
    dummy_input = torch.randn(16, 3, 224, 224)
    torch.onnx.export(model, dummy_input, "senet154.onnx", input_names=input_names, output_names=output_names,
                      opset_version=11)


if __name__ == "__main__":
    convert(sys.argv[1])