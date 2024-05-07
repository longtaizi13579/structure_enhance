from abc import ABC, abstractmethod


class BaseRunner(ABC):
    def __init__(self):
        self.model = None
        self.tokenizer = None

    @abstractmethod
    def prepare_dataset(self, args, split='train'):
        pass

    def reload_train_dataset_per_epoch(self, args, epoch):
        return None

    @abstractmethod
    def forward(self, model, payload, tokenizer, device):
        pass

    def before_inference(self, model, tokenizer, device, result_dict):
        pass

    @abstractmethod
    def inference(self, model, payload, tokenizer, device, result_dict):
        pass

    @staticmethod
    @abstractmethod
    def init_losses_dict():
        pass

    @staticmethod
    @abstractmethod
    def init_results_dict():
        pass

    @staticmethod
    @abstractmethod
    def measure_loss(payload, model_output, losses):
        pass

    @staticmethod
    @abstractmethod
    def init_meters_dict():
        pass

    @staticmethod
    @abstractmethod
    def measure_result(result_dict, meters):
        pass

    @staticmethod
    def score_func(meters):
        score = 0
        for key, meter in meters.items():
            score += meter.avg
        return score
