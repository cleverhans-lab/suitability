import torch
import numpy as np
import torch.nn.functional as F

'''
This module contains classes for evaluating signals on a sample level. Signals are used to evaluate the suitability of a model for a given task.
'''

class SampleSignal:
    '''
    Base class for evaluating signals on a sample level.
    '''
    def __init__(self, model, dataloader, device):
        self.model = model
        self.dataloader = dataloader
        self.device = device

    def evaluate(self):
        raise NotImplementedError("Subclasses should implement this method")
    
class ConfidenceSignal(SampleSignal):
    '''
    Signal that evaluates the confidence of a model on a sample level.
    '''
    def evaluate(self):
        self.model.eval()
        confidences = []
        with torch.no_grad():
            for data in self.dataloader:
                inputs, _ = data
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                confidences.append(F.softmax(outputs, dim=1).max(dim=1)[0].cpu().numpy())
        return np.concatenate(confidences)
    
class DecisionBoundarySignal(SampleSignal):
    def __init__(self, model, dataloader, device, max_steps, metric, perturbation):
        super().__init__(model, dataloader, device)
        self.max_steps = max_steps
        self.metric = metric
        self.perturbation = perturbation

    def evaluate(self):
        assert self.metric in ["l1", "l2", "linf"], f"Metric {self.metric} not supported, pick one of ['l1', 'l2', 'linf']"
        self.model.eval()
        
