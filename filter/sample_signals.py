import numpy as np
import torch
import torch.nn.functional as F

"""
This module contains classes for evaluating signals on a sample level. Signals are used to evaluate the suitability of a model for a given task.
"""


class SampleSignal:
    """
    Base class for evaluating signals on a sample level.
    """

    def __init__(self, model, dataloader, device):
        self.model = model
        self.dataloader = dataloader
        self.device = device

    def evaluate(self):
        raise NotImplementedError("Subclasses should implement this method")


class ConfidenceSignal(SampleSignal):
    """
    Signal that evaluates the confidence of a model on a sample level.
    """

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


class LogitSignal(SampleSignal):
    """
    Signal that evaluates the logits of a model on a sample level.
    This class directly uses the logits output by the model without applying softmax.
    """

    def __init__(self, model, dataloader, device):
        super().__init__(model, dataloader, device)

    def evaluate(self):
        self.model.eval()
        logits_list = []

        with torch.no_grad():
            for inputs, labels in self.dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)  # Direct logits output

                # Collect logits for each sample (detach them from the computation graph)
                logits_batch = outputs.cpu().numpy()
                logits_list.extend(logits_batch)

        # Return logits as a numpy array, correctness as a boolean array
        return np.array(logits_list)


class DecisionBoundarySignal(SampleSignal):
    """
    Signal that evaluates the distance of a sample to the decision boundary of a model.
    Loosely based on https://github.com/cleverhans-lab/dataset-inference/blob/main/src/attacks.py.
    """

    def __init__(
        self,
        model,
        dataloader,
        device,
        max_steps,
        metric,
        perturbation,
        noise_consistent=False,
    ):
        super().__init__(model, dataloader, device)
        self.max_steps = max_steps
        self.metric = metric
        self.perturbation = perturbation
        self.noise_consistent = noise_consistent

    def evaluate(self):
        assert self.metric in [
            "l1",
            "l2",
            "linf",
        ], f"Metric {self.metric} not supported, pick one of ['l1', 'l2', 'linf']"

        self.model.eval()
        max_mag_per_image = np.array([])

        # Define noise
        def l1_noise(X):
            return (
                torch.from_numpy(
                    np.random.laplace(
                        loc=0.0, scale=2 * self.perturbation, size=X.shape
                    )
                )
                .float()
                .to(self.device)
            )

        def l2_noise(X):
            return torch.normal(0, self.perturbation, size=X.shape).to(self.device)

        def linf_noise(X):
            return (
                torch.empty_like(X)
                .uniform_(-self.perturbation, self.perturbation)
                .to(self.device)
            )

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

                    inputs_r = inputs[remaining]

                    if self.noise_consistent:
                        dlt_r = i * base_delta[remaining]
                    else:
                        dlt_r = i * noise[self.metric](inputs_r)

                    preds_r = self.model(inputs_r + dlt_r)

                    remaining_r = (
                        torch.argmax(preds_r.data, 1) == predictions[remaining]
                    )
                    remaining_copy = remaining.clone()
                    remaining[remaining_copy] = remaining_r.clone()

                    max_step[remaining] = i

                max_mag_per_image = np.append(max_mag_per_image, max_step.cpu().numpy())

        return max_mag_per_image


class TrainingDynamicsSignal_Basic(SampleSignal):
    """
    Signal that evaluates the training dynamics of a model on a sample level.
    It does so by calculating the number of unique predictions made by the model across different checkpoints (training epochs).
    """

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

            sample_idx = 0  # Index of the current sample
            with torch.no_grad():
                for data in self.dataloader:
                    inputs, _ = data
                    inputs = inputs.to(self.device)
                    outputs = model(inputs)
                    predictions = torch.argmax(outputs, dim=1).cpu().numpy()

                    all_predictions[i, sample_idx : sample_idx + len(predictions)] = (
                        predictions
                    )
                    sample_idx += len(predictions)

        # Calculate the number of unique predictions for each sample
        for i in range(num_samples):
            unique_predictions[i] = len(np.unique(all_predictions[:, i]))

        return unique_predictions


class TrainingDynamicsSignal(SampleSignal):
    """
    Signal that evaluates the training dynamics of a model on a sample level.
    It calculates the prediction disagreement score across different checkpoints.
    Based on the "SPTD for classification" algorithm introduced in https://openreview.net/pdf?id=flg9EB6ikY.
    """

    def __init__(
        self, model, dataloader, device, model_checkpoints, k=2, max_epochs=None
    ):
        super().__init__(model, dataloader, device)
        self.model_checkpoints = model_checkpoints
        self.k = k
        self.max_epochs = max_epochs
        if self.max_epochs is None:
            eps = [
                int(checkpoint.split("/")[-1].split("_")[-1].split(".")[0])
                for checkpoint in model_checkpoints
            ]
            self.max_epochs = max(eps)

    def evaluate(self):
        num_samples = len(self.dataloader.dataset)
        num_models = len(self.model_checkpoints)

        all_predictions = np.zeros((num_models, num_samples), dtype=int)
        final_predictions = np.zeros(num_samples, dtype=int)
        disagreement_scores = np.zeros(num_samples, dtype=float)
        all_t = []
        for checkpoint in self.model_checkpoints:
            all_t.append(int(checkpoint.split("/")[-1].split("_")[-1].split(".")[0]))
        t = np.array(all_t)

        # Load the final model checkpoint
        final_model_checkpoint = self.model_checkpoints[-1]
        model = self.model
        model.load_state_dict(torch.load(final_model_checkpoint))
        model.to(self.device)
        model.eval()

        # Make predictions for the final model
        sample_idx = 0
        with torch.no_grad():
            for data in self.dataloader:
                inputs, _ = data
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()

                final_predictions[sample_idx : sample_idx + len(predictions)] = (
                    predictions
                )
                sample_idx += len(predictions)

        # Load each model checkpoint and calculate prediction disagreements
        for mod, checkpoint in enumerate(self.model_checkpoints[:-1]):
            model.load_state_dict(torch.load(checkpoint))
            model.to(self.device)
            model.eval()

            sample_idx = 0
            with torch.no_grad():
                for data in self.dataloader:
                    inputs, _ = data
                    inputs = inputs.to(self.device)
                    outputs = model(inputs)
                    predictions = torch.argmax(outputs, dim=1).cpu().numpy()

                    all_predictions[mod, sample_idx : sample_idx + len(predictions)] = (
                        predictions
                    )
                    sample_idx += len(predictions)

            # Calculate the disagreement score for each sample
            t = all_t[mod]
            at = all_predictions[mod, :] != final_predictions
            vt = (t / self.max_epochs) ** self.k
            disagreement_scores += vt * at

        return disagreement_scores
