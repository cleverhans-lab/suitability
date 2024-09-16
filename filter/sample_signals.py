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
    def __init__(self, model, dataloader, device):
        super().__init__(model, dataloader, device)

    def evaluate(self):
        self.model.eval()
        confidences = []
        correctness = []

        with torch.no_grad():
            for inputs, labels in self.dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                
                # Get confidence scores using softmax and taking the max
                softmax_outputs = F.softmax(outputs, dim=1)
                confidences_batch = softmax_outputs.max(dim=1)[0].cpu().numpy()
                
                # Calculate correctness (True for correct predictions, False for incorrect)
                predictions = torch.argmax(outputs, dim=1)
                correctness_batch = predictions.eq(labels).cpu().numpy()

                confidences.extend(confidences_batch)
                correctness.extend(correctness_batch)

        return np.array(confidences), np.array(correctness, dtype=bool)
    

class DecisionBoundarySignal(SampleSignal):
    '''
    Signal that evaluates the distance of a sample to the decision boundary of a model.
    Loosely based on https://github.com/cleverhans-lab/dataset-inference/blob/main/src/attacks.py.
    '''
    def __init__(self, model, dataloader, device, max_steps, metric, perturbation, noise_consistent=False):
        super().__init__(model, dataloader, device)
        self.max_steps = max_steps
        self.metric = metric
        self.perturbation = perturbation
        self.noise_consistent = noise_consistent

    def evaluate(self):
        assert self.metric in ["l1", "l2", "linf"], f"Metric {self.metric} not supported, pick one of ['l1', 'l2', 'linf']"

        self.model.eval()
        max_mag_per_image = np.array([])

        # Define noise
        l1_noise = lambda X: torch.from_numpy(np.random.laplace(loc=0.0, scale=2*self.perturbation, size=X.shape)).float().to(self.device) 
        l2_noise = lambda X: torch.normal(0, self.perturbation, size=X.shape).to(self.device)
        linf_noise = lambda X: torch.empty_like(X).uniform_(-self.perturbation, self.perturbation).to(self.device)
        noise = {"l1": l1_noise, "l2": l2_noise, "linf": linf_noise}
        
        with torch.no_grad():  # No need to calculate gradients for evaluation
            for data in self.dataloader:
                inputs, _ = data
                inputs = inputs.to(self.device)
                predictions = torch.argmax(self.model(inputs), dim=1)

                max_step = torch.zeros(inputs.size(0)).to(self.device)
                remaining = torch.ones(inputs.size(0)).to(self.device).bool()
                delta = noise[self.metric](inputs)
                base_delta = delta.clone()

                for i in range(1, self.max_steps):
                    if remaining.sum() == 0:
                        break

                    # Apply perturbation only to the remaining inputs
                    inputs_r = inputs[remaining]
                    delta_r = delta[remaining]

                    preds = self.model(inputs_r + delta_r)
                    new_remaining = (torch.argmax(preds.data, 1) == predictions[remaining])
                    
                    remaining[remaining] = new_remaining
                    max_step[remaining] = i

                    # Update delta only for the remaining inputs
                    if self.noise_consistent:
                        delta_r = (i+1) * base_delta[remaining]
                    else:
                        delta_r = (i+1) * noise[self.metric](inputs_r)
                    delta[remaining] = delta_r
                
                max_mag_per_image = np.append(max_mag_per_image, max_step.cpu().numpy())

        return max_mag_per_image
    

class TrainingDynamicsSignal(SampleSignal):
    '''
    Signal that evaluates the training dynamics of a model on a sample level.
    It does so by calculating the number of unique predictions made by the model across different checkpoints (training epochs).
    '''
    def __init__(self, model, dataloader, device, model_checkpoints):
        super().__init__(model, dataloader, device)
        self.model_checkpoints = model_checkpoints

    def evaluate(self):
        num_samples = len(self.dataloader.dataset)
        num_models = len(self.model_checkpoints)

        all_predictions = np.zeros((num_models, num_samples), dtype=int)
        unique_predictions = np.zeros(num_samples, dtype=int)

        # Load each model checkpoint and make predictions on the dataset
        for i, checkpoint in enumerate(self.model_checkpoints):
            model = self.model
            model.load_state_dict(torch.load(checkpoint))
            model.to(self.device)
            model.eval()

            sample_idx = 0 # Index of the current sample
            with torch.no_grad():
                for data in self.dataloader:
                    inputs, _ = data
                    inputs = inputs.to(self.device)
                    outputs = model(inputs)
                    predictions = torch.argmax(outputs, dim=1).cpu().numpy()

                    all_predictions[i, sample_idx:sample_idx+len(predictions)] = predictions
                    sample_idx += len(predictions)

        # Calculate the number of unique predictions for each sample
        for i in range(num_samples):
            unique_predictions[i] = len(np.unique(all_predictions[:, i]))

        return unique_predictions

        
