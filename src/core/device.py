import os
import torch


class DeviceManager:
    def __init__(self):
        self.device = self._detect_device()

    def _detect_device(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.backends.cudnn.benchmark = True
            print(f"[Device] CUDA — {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("[Device] Apple MPS")
        else:
            device = torch.device("cpu")
            print("[Device] CPU")
        return device

    @property
    def is_cuda(self):
        return self.device.type == "cuda"

    @property
    def is_mps(self):
        return self.device.type == "mps"

    def get_optimal_workers(self):
        cores = os.cpu_count() or 1
        if os.name == "nt":
            return min(4, max(1, cores - 1))
        return min(8, max(1, cores - 1))

    def get_scaler(self):
        if self.is_cuda:
            return torch.amp.GradScaler("cuda")
        return None
