import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression

class DifferenceOfEntropy:
    def __init__(self, model, reference_data, device, num_shifts=100):
        """
        Initialize DoE with a model, reference data, and device.
        Learns a linear relationship between mean entropy differences and performance differences
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

        # Calculate mean entropy and accuracy on reference data
        all_entropies, all_correct = self._calculate_entropy_and_accuracy(reference_data)
        self.reference_entropy_mean = np.mean(all_entropies)
        self.reference_accuracy = np.mean(all_correct)

        # Generate synthetic shifts and fit linear regression
        self.regressor = self._fit_regression_on_shifts()

    def _calculate_entropy_and_accuracy(self, data_loader):
        """
        Calculate entropies and correctness for given data.

        data_loader: DataLoader with labeled data.
        Returns: Entropy scores and correctness labels.
        """
        all_entropies = []
        all_correct = []

        with torch.no_grad():
            for batch in data_loader:
                inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
                outputs = self.model(inputs)

                # Store correctness for accuracy calculation
                correct = (torch.argmax(outputs, dim=1) == labels).cpu().numpy()
                all_correct.extend(correct)

                # Calculate entropy scores
                probabilities = F.softmax(outputs, dim=1)
                entropies = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1).cpu().numpy()
                all_entropies.extend(entropies)

        return np.array(all_entropies), np.array(all_correct)

    def _fit_regression_on_shifts(self):
        """
        Generate synthetic shifts, compute entropy and accuracy differences,
        and fit linear regression model.

        Returns: Fitted LinearRegression model.
        """
        mean_entropy_diffs = []
        accuracy_diffs = []

        all_entropies, all_correct = self._calculate_entropy_and_accuracy(self.reference_data)

        for _ in range(self.num_shifts):
            random_indices = np.random.choice(
                len(all_correct), size=len(all_correct) // 2, replace=True
            )
            shifted_entropies = all_entropies[random_indices]
            shifted_correct = all_correct[random_indices]

            shifted_entropy_mean, shifted_accuracy = (
                np.mean(shifted_entropies),
                np.mean(shifted_correct),
            )

            # Compute differences in entropy and accuracy relative to reference
            mean_entropy_diffs.append(shifted_entropy_mean - self.reference_entropy_mean)
            accuracy_diffs.append(shifted_accuracy - self.reference_accuracy)

        # Fit linear regression with entropy differences as input and accuracy differences as target
        regressor = LinearRegression().fit(
            np.array(mean_entropy_diffs).reshape(-1, 1), accuracy_diffs
        )
        return regressor

    def estimate_performance(self, user_data=None, user_entropies=None):
        """
        Estimate the performance on new data based on the learned relationship.

        user_data: DataLoader with new data to evaluate.
        user_entropies: List of entropy scores for the user data.
        Returns: Estimated performance metric.
        """
        if user_data is None and user_entropies is None:
            raise ValueError("Either user_data or user_entropies must be provided.")
        elif user_data is not None and user_entropies is not None:
            raise ValueError("Only one of user_data or user_entropies should be provided.")

        # If user_data is provided, calculate the entropies
        if user_data is not None:
            eval_entropies, _ = self._calculate_entropy_and_accuracy(user_data)
        else:
            eval_entropies = user_entropies

        eval_entropy_mean = np.mean(eval_entropies)

        # Predict performance difference using the trained regressor
        entropy_diff = eval_entropy_mean - self.reference_entropy_mean
        performance_diff = self.regressor.predict([[entropy_diff]])[0]

        # Estimated performance on new data
        estimated_performance = self.reference_accuracy + performance_diff
        return estimated_performance

    def binary_evaluation(self, user_data=None, user_entropies=None):
        """
        Evaluate new user data by comparing its estimated performance to the reference data's performance.

        user_data: DataLoader with new data to evaluate.
        user_entropies: List of entropy scores for the user data.
        Returns: Boolean indicating if the estimated performance exceeds the reference performance.
        """
        estimated_performance = self.estimate_performance(user_data, user_entropies)
        return estimated_performance >= self.reference_accuracy
