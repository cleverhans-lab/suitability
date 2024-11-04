import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression


class DifferenceOfConfidence:
    def __init__(self, model, reference_data, device, num_shifts=100):
        """
        Initialize DoC with a model, reference data, and device.
        Learns a linear relationship between mean confidence differences and performance differences
        across synthetic distribution shifts.

        model: PyTorch model with classifier.
        reference_data: DataLoader with labeled reference data.
        device: Device to run computations on (e.g., 'cpu' or 'cuda').
        num_shifts: Number of random re-samplings to generate synthetic shifts.
        """
        self.model = model
        self.reference_data = reference_data
        self.device = device
        self.num_shifts = num_shifts
        self.model.eval()

        # Calculate max confidence mean and accuracy on reference data
        all_confidences, all_correct = self._calculate_confidence_and_accuracy(
            reference_data
        )
        self.reference_conf_mean = np.mean(all_confidences)
        self.reference_accuracy = np.mean(all_correct)

        # Generate synthetic shifts and fit linear regression
        self.regressor = self._fit_regression_on_shifts()

    def _calculate_confidence_and_accuracy(self, data_loader):
        """
        Calculate max confidences and correctness for given data.

        data_loader: DataLoader with labeled data.
        Returns: Confidence scores and correctness labels.
        """
        all_confidences = []
        all_correct = []

        with torch.no_grad():
            for batch in data_loader:
                inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
                outputs = self.model(inputs)

                # Store correctness for accuracy calculation
                correct = (torch.argmax(outputs, dim=1) == labels).cpu().numpy()
                all_correct.extend(correct)

                # Calculate confidence scores using maximum confidence
                max_confidences = F.softmax(outputs, dim=1).max(dim=1)[0].cpu().numpy()
                all_confidences.extend(max_confidences)

        return np.array(all_confidences), np.array(all_correct)

    def _fit_regression_on_shifts(self):
        """
        Generate synthetic shifts, compute confidence and accuracy differences,
        and fit linear regression model.

        Returns: Fitted LinearRegression model.
        """
        mean_conf_diffs = []
        accuracy_diffs = []

        all_confidences, all_correct = self._calculate_confidence_and_accuracy(
            self.reference_data
        )

        for _ in range(self.num_shifts):
            random_indices = np.random.choice(
                len(all_correct), size=len(all_correct) // 2, replace=True
            )
            shifted_confidences = all_confidences[random_indices]
            shifted_correct = all_correct[random_indices]

            shifted_conf_mean, shifted_accuracy = (
                np.mean(shifted_confidences),
                np.mean(shifted_correct),
            )

            # Compute differences in confidence and accuracy relative to reference
            mean_conf_diffs.append(shifted_conf_mean - self.reference_conf_mean)
            accuracy_diffs.append(shifted_accuracy - self.reference_accuracy)

        # Fit linear regression with confidence differences as input and accuracy differences as target
        regressor = LinearRegression().fit(
            np.array(mean_conf_diffs).reshape(-1, 1), accuracy_diffs
        )
        return regressor

    def estimate_performance(self, user_data):
        """
        Estimate the performance on new data based on the learned relationship.

        evaluation_data: DataLoader with new data to evaluate.
        Returns: Estimated performance metric.
        """
        eval_conf_all, _ = self._calculate_confidence_and_accuracy(user_data)

        eval_conf_mean = np.mean(eval_conf_all)

        # Predict performance difference using the trained regressor
        confidence_diff = eval_conf_mean - self.reference_conf_mean
        performance_diff = self.regressor.predict([[confidence_diff]])[0]

        # Estimated performance on new data
        estimated_performance = self.reference_accuracy + performance_diff
        return estimated_performance

    def binary_evaluation(self, user_data):
        """
        Evaluate new user data by comparing its estimated performance to the reference data's performance.

        user_data: DataLoader with new data to evaluate.
        Returns: Boolean indicating if the estimated performance exceeds the reference performance.
        """
        estimated_performance = self.estimate_performance(user_data)
        return estimated_performance >= self.reference_accuracy
