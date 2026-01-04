# -*- coding: utf-8 -*-
import math
import numpy as np
import torch
import torch.nn as nn


class BMN(nn.Module):
    def __init__(self, opt):
        super(BMN, self).__init__()
        self.tscale = opt["temporal_scale"]
        self.prop_boundary_ratio = opt["prop_boundary_ratio"]
        self.num_sample = opt["num_sample"]
        self.num_sample_perbin = opt["num_sample_perbin"]
        self.feat_dim=opt["feat_dim"]

        self.hidden_dim_1d = 256
        self.hidden_dim_2d = 128
        self.hidden_dim_3d = 512

        self._get_interp1d_mask()

        # Base Module
        self.x_1d_b = nn.Sequential(
            nn.Conv1d(self.feat_dim, self.hidden_dim_1d, kernel_size=3, padding=1, groups=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1, groups=1),
            nn.ReLU(inplace=True)
        )

        # Temporal Evaluation Module
        self.x_1d_s = nn.Sequential(
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1, groups=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.x_1d_e = nn.Sequential(
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1, groups=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Proposal Evaluation Module
        self.x_1d_p = nn.Sequential(
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.x_3d_p = nn.Sequential(
            nn.Conv3d(self.hidden_dim_1d, self.hidden_dim_3d, kernel_size=(self.num_sample, 1, 1),stride=(self.num_sample, 1, 1)),
            nn.ReLU(inplace=True)
        )
        self.x_2d_p = nn.Sequential(
            nn.Conv2d(self.hidden_dim_3d, self.hidden_dim_2d, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim_2d, self.hidden_dim_2d, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim_2d, self.hidden_dim_2d, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim_2d, 2, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        base_feature = self.x_1d_b(x)
        start = self.x_1d_s(base_feature).squeeze(1)
        end = self.x_1d_e(base_feature).squeeze(1)
        confidence_map = self.x_1d_p(base_feature)
        confidence_map = self._boundary_matching_layer(confidence_map)
        confidence_map = self.x_3d_p(confidence_map).squeeze(2)
        confidence_map = self.x_2d_p(confidence_map)
        return confidence_map, start, end

    def _boundary_matching_layer(self, x):
        input_size = x.size()
        out = torch.matmul(x, self.sample_mask).reshape(input_size[0],input_size[1],self.num_sample,self.tscale,self.tscale)
        return out

    def _get_interp1d_bin_mask(self, seg_xmin, seg_xmax, tscale, num_sample, num_sample_perbin):
        # generate sample mask for a boundary-matching pair
        plen = float(seg_xmax - seg_xmin)
        plen_sample = plen / (num_sample * num_sample_perbin - 1.0)
        total_samples = [
            seg_xmin + plen_sample * ii
            for ii in range(num_sample * num_sample_perbin)
        ]
        p_mask = []
        for idx in range(num_sample):
            bin_samples = total_samples[idx * num_sample_perbin:(idx + 1) * num_sample_perbin]
            bin_vector = np.zeros([tscale])
            for sample in bin_samples:
                sample_upper = math.ceil(sample)
                sample_decimal, sample_down = math.modf(sample)
                if int(sample_down) <= (tscale - 1) and int(sample_down) >= 0:
                    bin_vector[int(sample_down)] += 1 - sample_decimal
                if int(sample_upper) <= (tscale - 1) and int(sample_upper) >= 0:
                    bin_vector[int(sample_upper)] += sample_decimal
            bin_vector = 1.0 / num_sample_perbin * bin_vector
            p_mask.append(bin_vector)
        p_mask = np.stack(p_mask, axis=1)
        return p_mask

    def _get_interp1d_mask(self):
        # generate sample mask for each point in Boundary-Matching Map
        mask_mat = []
        for end_index in range(self.tscale):
            mask_mat_vector = []
            for start_index in range(self.tscale):
                if start_index <= end_index:
                    p_xmin = start_index
                    p_xmax = end_index + 1
                    center_len = float(p_xmax - p_xmin) + 1
                    sample_xmin = p_xmin - center_len * self.prop_boundary_ratio
                    sample_xmax = p_xmax + center_len * self.prop_boundary_ratio
                    p_mask = self._get_interp1d_bin_mask(
                        sample_xmin, sample_xmax, self.tscale, self.num_sample,
                        self.num_sample_perbin)
                else:
                    p_mask = np.zeros([self.tscale, self.num_sample])
                mask_mat_vector.append(p_mask)
            mask_mat_vector = np.stack(mask_mat_vector, axis=2)
            mask_mat.append(mask_mat_vector)
        mask_mat = np.stack(mask_mat, axis=3)
        mask_mat = mask_mat.astype(np.float32)
        self.sample_mask = nn.Parameter(torch.Tensor(mask_mat).view(self.tscale, -1), requires_grad=False)


if __name__ == '__main__':
    import opts
    opt = opts.parse_opt()
    opt = vars(opt)
    model=BMN(opt)
    input=torch.randn(2,400,100)
    a,b,c=model(input)
    print(a.shape,b.shape,c.shape)

# import math
# import numpy as np
# import torch
# import torch.nn as nn
# from functools import partial

# class BMN(nn.Module):
#     def __init__(self, opt):
#         super(BMN, self).__init__()
#         self.tscale = opt["temporal_scale"]
#         self.prop_boundary_ratio = opt["prop_boundary_ratio"]
#         self.num_sample = opt["num_sample"]
#         self.num_sample_perbin = opt["num_sample_perbin"]
#         self.feat_dim=opt["feat_dim"]

#         self.hidden_dim_1d = 256
#         self.hidden_dim_2d = 128
#         self.hidden_dim_3d = 512

#         self._get_interp1d_mask()

#         # Base Module
#         self.x_1d_b = nn.Sequential(
#             nn.Conv1d(self.feat_dim, self.hidden_dim_1d, kernel_size=3, padding=1, groups=1),
#             nn.ReLU(inplace=False),  # 改为False
#             nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1, groups=1),
#             nn.ReLU(inplace=False)   # 改为False
#         )

#         # Temporal Evaluation Module
#         self.x_1d_s = nn.Sequential(
#             nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1, groups=1),
#             nn.ReLU(inplace=False),  # 改为False
#             nn.Conv1d(self.hidden_dim_1d, 1, kernel_size=1),
#             nn.Sigmoid()
#         )
#         self.x_1d_e = nn.Sequential(
#             nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1, groups=1),
#             nn.ReLU(inplace=False),  # 改为False
#             nn.Conv1d(self.hidden_dim_1d, 1, kernel_size=1),
#             nn.Sigmoid()
#         )

#         # Proposal Evaluation Module
#         self.x_1d_p = nn.Sequential(
#             nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1),
#             nn.ReLU(inplace=False)   # 改为False
#         )
#         self.x_3d_p = nn.Sequential(
#             nn.Conv3d(self.hidden_dim_1d, self.hidden_dim_3d, kernel_size=(self.num_sample, 1, 1),stride=(self.num_sample, 1, 1)),
#             nn.ReLU(inplace=False)   # 改为False
#         )
#         self.x_2d_p = nn.Sequential(
#             nn.Conv2d(self.hidden_dim_3d, self.hidden_dim_2d, kernel_size=1),
#             nn.ReLU(inplace=False),  # 改为False
#             nn.Conv2d(self.hidden_dim_2d, self.hidden_dim_2d, kernel_size=3, padding=1),
#             nn.ReLU(inplace=False),  # 改为False
#             nn.Conv2d(self.hidden_dim_2d, self.hidden_dim_2d, kernel_size=3, padding=1),
#             nn.ReLU(inplace=False),  # 改为False
#             nn.Conv2d(self.hidden_dim_2d, 2, kernel_size=1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         base_feature = self.x_1d_b(x)
#         start = self.x_1d_s(base_feature).squeeze(1)
#         end = self.x_1d_e(base_feature).squeeze(1)
#         confidence_map = self.x_1d_p(base_feature)
#         confidence_map = self._boundary_matching_layer(confidence_map)
#         confidence_map = self.x_3d_p(confidence_map).squeeze(2)
#         confidence_map = self.x_2d_p(confidence_map)
#         return confidence_map, start, end

#     def _boundary_matching_layer(self, x):
#         input_size = x.size()
#         out = torch.matmul(x, self.sample_mask).reshape(input_size[0],input_size[1],self.num_sample,self.tscale,self.tscale)
#         return out

#     def _get_interp1d_bin_mask(self, seg_xmin, seg_xmax, tscale, num_sample, num_sample_perbin):
#         # generate sample mask for a boundary-matching pair
#         plen = float(seg_xmax - seg_xmin)
#         plen_sample = plen / (num_sample * num_sample_perbin - 1.0)
#         total_samples = [
#             seg_xmin + plen_sample * ii
#             for ii in range(num_sample * num_sample_perbin)
#         ]
#         p_mask = []
#         for idx in range(num_sample):
#             bin_samples = total_samples[idx * num_sample_perbin:(idx + 1) * num_sample_perbin]
#             bin_vector = np.zeros([tscale])
#             for sample in bin_samples:
#                 sample_upper = math.ceil(sample)
#                 sample_decimal, sample_down = math.modf(sample)
#                 if int(sample_down) <= (tscale - 1) and int(sample_down) >= 0:
#                     bin_vector[int(sample_down)] += 1 - sample_decimal
#                 if int(sample_upper) <= (tscale - 1) and int(sample_upper) >= 0:
#                     bin_vector[int(sample_upper)] += sample_decimal
#             bin_vector = 1.0 / num_sample_perbin * bin_vector
#             p_mask.append(bin_vector)
#         p_mask = np.stack(p_mask, axis=1)
#         return p_mask

#     def _get_interp1d_mask(self):
#         # generate sample mask for each point in Boundary-Matching Map
#         mask_mat = []
#         for end_index in range(self.tscale):
#             mask_mat_vector = []
#             for start_index in range(self.tscale):
#                 if start_index <= end_index:
#                     p_xmin = start_index
#                     p_xmax = end_index + 1
#                     center_len = float(p_xmax - p_xmin) + 1
#                     sample_xmin = p_xmin - center_len * self.prop_boundary_ratio
#                     sample_xmax = p_xmax + center_len * self.prop_boundary_ratio
#                     p_mask = self._get_interp1d_bin_mask(
#                         sample_xmin, sample_xmax, self.tscale, self.num_sample,
#                         self.num_sample_perbin)
#                 else:
#                     p_mask = np.zeros([self.tscale, self.num_sample])
#                 mask_mat_vector.append(p_mask)
#             mask_mat_vector = np.stack(mask_mat_vector, axis=2)
#             mask_mat.append(mask_mat_vector)
#         mask_mat = np.stack(mask_mat, axis=3)
#         mask_mat = mask_mat.astype(np.float32)
#         self.sample_mask = nn.Parameter(torch.Tensor(mask_mat).view(self.tscale, -1), requires_grad=False)

#     def register_debug_hooks(self):
#         """
#         Register forward and backward hooks for debugging convergence issues
#         """
#         def observe_input_output_forward(module, module_input, module_output, name):
#             if not hasattr(module, 'weight'):
#                 return

#             input = module_input[0]
#             output = module_output

#             with torch.no_grad():
#                 # 对于大型张量，只检查一小部分元素
#                 if input.numel() > 1000000:  # 如果元素数量超过100万
#                     # 随机采样一小部分元素进行检查
#                     input_sample = input.flatten()[::max(1, input.numel()//100000)]
#                     if torch.isnan(input_sample).any() or torch.isinf(input_sample).any():
#                         print(f"ERROR:::forward:::{name}:::input:::contains NaN/Inf")
#                     else:
#                         m = input_sample.float().abs().max().item()
#                         print(f"observe_input_output:::forward:::{name}:::input:::max::{m}")
#                 else:
#                     if torch.isnan(input).any() or torch.isinf(input).any():
#                         print(f"ERROR:::forward:::{name}:::input:::contains NaN/Inf")
#                     else:
#                         m = input.float().abs().max().item()
#                         print(f"observe_input_output:::forward:::{name}:::input:::max::{m}")
                
#                 # 对输出也采用同样的策略
#                 if output.numel() > 1000000:
#                     output_sample = output.flatten()[::max(1, output.numel()//100000)]
#                     if torch.isnan(output_sample).any() or torch.isinf(output_sample).any():
#                         print(f"ERROR:::forward:::{name}:::output:::contains NaN/Inf")
#                     else:
#                         o = output_sample.float().abs().max().item()
#                         print(f"observe_input_output:::forward:::{name}:::output:::max::{o}")
#                 else:
#                     if torch.isnan(output).any() or torch.isinf(output).any():
#                         print(f"ERROR:::forward:::{name}:::output:::contains NaN/Inf")
#                     else:
#                         o = output.float().abs().max().item()
#                         print(f"observe_input_output:::forward:::{name}:::output:::max::{o}")

#         def observe_input_output_backward(module, module_gradinput, module_gradoutput, name):
#             if not hasattr(module, "weight") or not hasattr(module.weight, 'grad'):
#                 return

#             gradinput = module_gradinput[0] if module_gradinput else None
#             gradoutput = module_gradoutput[0] if module_gradoutput else None
#             weightgrad = module.weight.grad

#             try:
#                 with torch.no_grad():
#                     # 对梯度也采用采样策略
#                     if gradinput is not None:
#                         if gradinput.numel() > 1000000:
#                             gradinput_sample = gradinput.flatten()[::max(1, gradinput.numel()//100000)]
#                             if torch.isnan(gradinput_sample).any() or torch.isinf(gradinput_sample).any():
#                                 print(f"ERROR:::backward:::{name}:::gradinput:::contains NaN/Inf")
#                             else:
#                                 print(f"observe_input_output:::backward:::{name}:::gradinput:::max::{gradinput_sample.abs().max().item()}")
#                         else:
#                             if torch.isnan(gradinput).any() or torch.isinf(gradinput).any():
#                                 print(f"ERROR:::backward:::{name}:::gradinput:::contains NaN/Inf")
#                             else:
#                                 print(f"observe_input_output:::backward:::{name}:::gradinput:::max::{gradinput.abs().max().item()}")
                    
#                     if gradoutput is not None:
#                         if gradoutput.numel() > 1000000:
#                             gradoutput_sample = gradoutput.flatten()[::max(1, gradoutput.numel()//100000)]
#                             if torch.isnan(gradoutput_sample).any() or torch.isinf(gradoutput_sample).any():
#                                 print(f"ERROR:::backward:::{name}:::gradoutput:::contains NaN/Inf")
#                             else:
#                                 print(f"observe_input_output:::backward:::{name}:::gradoutput:::max::{gradoutput_sample.abs().max().item()}")
#                         else:
#                             if torch.isnan(gradoutput).any() or torch.isinf(gradoutput).any():
#                                 print(f"ERROR:::backward:::{name}:::gradoutput:::contains NaN/Inf")
#                             else:
#                                 print(f"observe_input_output:::backward:::{name}:::gradoutput:::max::{gradoutput.abs().max().item()}")
                    
#                     if weightgrad is not None:
#                         if weightgrad.numel() > 1000000:
#                             weightgrad_sample = weightgrad.flatten()[::max(1, weightgrad.numel()//100000)]
#                             if torch.isnan(weightgrad_sample).any() or torch.isinf(weightgrad_sample).any():
#                                 print(f"ERROR:::backward:::{name}:::weightgrad:::contains NaN/Inf")
#                             else:
#                                 print(f"observe_input_output:::backward:::{name}:::weightgrad:::max::{weightgrad_sample.abs().max().item()}")
#                         else:
#                             if torch.isnan(weightgrad).any() or torch.isinf(weightgrad).any():
#                                 print(f"ERROR:::backward:::{name}:::weightgrad:::contains NaN/Inf")
#                             else:
#                                 print(f"observe_input_output:::backward:::{name}:::weightgrad:::max::{weightgrad.abs().max().item()}")
#             except Exception as e:
#                 print(f"Exception in backward hook for {name}: {e}")
#                 pass

#         # Register hooks only for convolutional and linear layers
#         for name, module in self.named_modules():
#             if len(list(module.children())) == 0 and ("conv" in str(module).lower() or "linear" in str(module).lower()):
#                 module.register_forward_hook(partial(
#                     observe_input_output_forward,
#                     name=name))
#                 module.register_full_backward_hook(partial(
#                     observe_input_output_backward,
#                     name=name))