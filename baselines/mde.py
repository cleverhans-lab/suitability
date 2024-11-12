import numpy as np
import torch
from sklearn.linear_model import LinearRegression

class MetaDistributionEnergy:
    def __init__(self, model, reference_data, device, T=1.0, num_shifts=100):
        """
        Initialize MDE with a model, reference data, device, temperature T, and num_shifts.
        Learns a linear relationship between avg_energies and accuracy across synthetic shifts.

        model: PyTorch model with classifier.
        reference_data: DataLoader with labeled reference data.
        device: Device to run computations on (e.g., 'cpu' or 'cuda').
        T: Temperature parameter for energy calculation.
        num_shifts: Number of random re-samplings to generate synthetic shifts.
        """
        self.model = model
        self.reference_data = reference_data
        self.device = device
        self.T = T
        self.num_shifts = num_shifts
        self.model.eval()

        # Generate synthetic shifts and fit linear regression
        self.regressor = self._fit_regression_on_shifts()

    def _calculate_energy_and_accuracy(self, data_loader):
        """
        Calculate energy values and correctness for given data.

        data_loader: DataLoader with labeled data.
        Returns: Energy values and correctness labels.
        """
        all_energies = []
        all_correct = []

        with torch.no_grad():
            for batch in data_loader:
                inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
                outputs = self.model(inputs)

                # Store correctness for accuracy calculation
                correct = (torch.argmax(outputs, dim=1) == labels).cpu().numpy()
                all_correct.extend(correct)

                # Calculate energy using MetaDistributionEnergy formula
                energies = -self.T * torch.logsumexp(outputs / self.T, dim=1)
                all_energies.extend(energies.cpu().numpy())

        return np.array(all_energies), np.array(all_correct)

    def _aggregate_energies(self, energies):
        """
        Aggregate energies across a dataset using log-softmax and mean.

        energies: Array of per-sample energies.
        Returns: Aggregated avg_energies score.
        """
        energies_tensor = torch.tensor(energies, device=self.device)
        avg_energies = -torch.log_softmax(energies_tensor, dim=0).mean().item()
        return avg_energies

    def _fit_regression_on_shifts(self):
        """
        Generate synthetic shifts, compute avg_energies and accuracy for each shift,
        and fit linear regression model.

        Returns: Fitted LinearRegression model.
        """
        avg_energies_list = []
        accuracy_list = []

        all_energies, all_correct = self._calculate_energy_and_accuracy(self.reference_data)
        self.reference_accuracy = np.mean(all_correct)

        for _ in range(self.num_shifts):
            # Randomly sample half of the data for a synthetic shift
            random_indices = np.random.choice(len(all_correct), size=len(all_correct) // 2, replace=True)
            shifted_energies = all_energies[random_indices]
            shifted_correct = all_correct[random_indices]

            # Calculate avg_energies and accuracy for the synthetic shift
            avg_energy = self._aggregate_energies(shifted_energies)
            accuracy = np.mean(shifted_correct)

            avg_energies_list.append(avg_energy)
            accuracy_list.append(accuracy)

        # Fit linear regression with avg_energies as input and accuracy as target
        regressor = LinearRegression().fit(np.array(avg_energies_list).reshape(-1, 1), accuracy_list)
        return regressor

    def estimate_performance(self, user_data=None, user_energies=None):
        """
        Estimate the performance on new data based on the learned relationship.

        user_data: DataLoader with new data to evaluate.
        user_energies: List of energy scores for the user data.
        Returns: Estimated performance metric.
        """
        if user_data is None and user_energies is None:
            raise ValueError("Either user_data or user_energies must be provided.")
        elif user_data is not None and user_energies is not None:
            raise ValueError("Only one of user_data or user_energies should be provided.")

        # If user_data is provided, calculate the energies
        if user_data is not None:
            eval_energies, _ = self._calculate_energy_and_accuracy(user_data)
        else:
            eval_energies = user_energies

        eval_avg_energy = self._aggregate_energies(eval_energies)

        # Predict accuracy using the trained regressor
        estimated_performance = self.regressor.predict([[eval_avg_energy]])[0]
        return estimated_performance

    def binary_evaluation(self, user_data=None, user_energies=None):
        """
        Evaluate new user data by comparing its estimated performance to a threshold.

        user_data: DataLoader with new data to evaluate.
        user_energies: List of energy scores for the user data.
        threshold: Accuracy threshold for binary evaluation.
        Returns: Boolean indicating if the estimated performance meets or exceeds the threshold.
        """
        estimated_performance = self.estimate_performance(user_data, user_energies)
        return estimated_performance >= self.reference_accuracy
