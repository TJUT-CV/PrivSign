import pynvml
import time
import threading
# --- 监控线程 ---
class GpuMonitor(threading.Thread):
    def __init__(self, device_index=0, delay=0.5):
        super(GpuMonitor, self).__init__()
        self.device_index = device_index
        self.delay = delay
        self.running = False
        self.utilization_records = []
        self.memory_records = []

        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)

    def run(self):
        self.running = True
        while self.running:
            utilization = pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)

            self.utilization_records.append(utilization)
            self.memory_records.append(mem_info.used / 1024 ** 2)  # in MiB
            time.sleep(self.delay)

    def stop(self):
        self.running = False
        pynvml.nvmlShutdown()

    # self.arg.phase == 'test':
    #         # 启动监控
    #         monitor = GpuMonitor(device_index=6, delay=0.1)  # 100ms 采样一次
    #         monitor.start()
    #         # ===============================================
    #         if self.arg.load_weights is None and self.arg.load_checkpoints is None:
    #             raise ValueError('Please appoint --load-weights.')
    #         self.recoder.print_log('Model:   {}.'.format(self.arg.model))
    #         self.recoder.print_log('Weights: {}.'.format(self.arg.load_weights))
    #         # train_wer = seq_eval(self.arg, self.data_loader["train_eval"], self.model, self.device,
    #         #                      "train", 6667, self.arg.work_dir, self.recoder, self.arg.evaluate_tool)
    #         dev_wer = seq_eval(self.arg, self.data_loader["dev"], self.model, self.device,
    #                            "dev", 6667, self.arg.work_dir, self.recoder, self.arg.evaluate_tool)
    #         test_wer = seq_eval(self.arg, self.data_loader["test"], self.model, self.device,
    #                             "test", 6667, self.arg.work_dir, self.recoder, self.arg.evaluate_tool)
    #         self.recoder.print_log('Evaluation Done.\n')
    #
    #         # ===============================================
    #
    #         # 停止监控
    #         monitor.stop()
    #         monitor.join()
    #
    #         # --- 分析结果 ---
    #         if monitor.utilization_records:
    #             avg_util = sum(monitor.utilization_records) / len(monitor.utilization_records)
    #             max_util = max(monitor.utilization_records)
    #             avg_mem = sum(monitor.memory_records) / len(monitor.memory_records)
    #             max_mem = max(monitor.memory_records)
    #
    #             self.recoder.print_log(f"\n--- 监控结果 ---")
    #             self.recoder.print_log(f"平均 GPU 利用率: {avg_util:.2f}%")
    #             self.recoder.print_log(f"峰值 GPU 利用率: {max_util:.2f}%")
    #             self.recoder.print_log(f"平均显存使用: {avg_mem:.2f} MiB")
    #             self.recoder.print_log(f"峰值显存使用: {max_mem:.2f} MiB")