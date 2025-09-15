import datetime

import torch
import torch.nn as nn
from torch import Tensor
from utils import Recorder, GpuDataParallel
import time
# from torchprofile import profile_macs
# from thop import profile
from fvcore.nn import FlopCountAnalysis, parameter_count, parameter_count_table, flop_count_str


# from ptflops import get_model_complexity_info


class ModelInspector:
    def __init__(self, model, device: GpuDataParallel, log_recorder: Recorder, file_path):
        self.model = model
        self.device = device
        self.log_recorder = log_recorder
        self.file_path = '{}/model_inspect.txt'.format(file_path)

    def calculate_parameters(self):
        """
        Calculate the parameters of the model
        """

        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # print("Model = %s" % str(self.model))
        params_str = "All model(include the teacher model and the student model)'s number of params (M): %.2f" % (n_parameters / 1.e6)
        self.log_recorder.print_log(str=params_str, path=self.file_path, print_time=True)
        return n_parameters

    def calculate_flops(self, input_size, seq_len):
        """
        Calculate the number of FLOPs of the model
        """
        vid = torch.randn(input_size)
        vid_lgt = torch.tensor([seq_len])
        inputs = (vid, vid_lgt)
        inputs = tuple(self.device.data_to_device(inp) for inp in inputs)
        # ----------------------------------torchprofile-------------------------------
        # macs = profile_macs(self.model, inputs)
        # print('macs: {}'.format(macs))
        # print('macs G: {}'.format(macs / 1.e9))
        # print("-----------------------------thop----------------------------")
        # macs, params = profile(self.model, inputs)
        # print(f"macs: {macs}")
        # print(f"Params: {params}")
        # print(f"model macs (G): {macs / 1.e9}")
        # print(f"model params (M):{params}, input_size:, {input_size}")
        # print("-----------------------------使用fvcore计算Params----------------------------")
        # params = parameter_count(self.model)
        # print(params[""])
        # self.log_recorder.print_log(str(params), path=self.file_path, print_time=True)
        print("-----------------------------使用fvcore计算FLOPS----------------------------")
        flop_counter = FlopCountAnalysis(self.model, inputs)
        flops = flop_counter.total()
        flops_g = flops / 1.e9  # Convert to GFLOPs
        flops_str = (f"fvcore: model flops: {flops} "
                     f"fvcore: model flops (G): {flops_g} "
                     f"input_size: {input_size}")
        self.log_recorder.print_log(flops_str, path=self.file_path, print_time=True)
        self.log_recorder.print_log(flop_count_str(flop_counter), path=self.file_path, print_time=True, is_print2screen=False)
        self.log_recorder.print_log(parameter_count_table(self.model), path=self.file_path, print_time=True, is_print2screen=False)
        return flops

    # def measure_training_time(self, train_func, *args, **kwargs):
    #     start_time = time.time()
    #     result = train_func(*args, **kwargs)
    #     total_time = time.time() - start_time
    #     total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    #     print('Training time {}'.format(total_time_str))
    #     return total_time, result

    # def measure_inference_time(self, data_loader, use_amp=False):
    #     self.model.eval()
    #     total_inference_time = 0
    #     total_samples = 0
    #     with torch.no_grad():
    #         for images, target in data_loader:
    #             images = images.to(self.device, non_blocking=True)
    #             target = target.to(self.device, non_blocking=True)
    #
    #             start_time = time.time()
    #             if use_amp:
    #                 with torch.cuda.amp.autocast():
    #                     output = self.model(images)
    #             else:
    #                 output = self.model(images)
    #             end_time = time.time()
    #
    #             inference_time = end_time - start_time
    #             total_inference_time += inference_time
    #             total_samples += images.size(0)
    #
    #     avg_inference_time = total_inference_time / total_samples
    #     print(f"Average inference time per sample: {avg_inference_time:.6f} seconds")
    #     return avg_inference_time


# 示例使用
def main(args):
    a = 1
