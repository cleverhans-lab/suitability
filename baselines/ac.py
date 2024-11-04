import numpy as np
import torch
import torch.nn.functional as F


class AverageConfidence:
    def __init__(self, model, test_data, device):
        """
        Initialize the AverageConfidence with a model, test dataset, and device.

        model: Trained PyTorch model used for predictions.
        test_data: DataLoader containing the test dataset.
        device: The device (e.g., 'cpu' or 'cuda') to perform computations on.
        """
        self.model = model
        self.test_data = test_data
        self.device = device

        self.model.eval()

        all_confidences = []

        with torch.no_grad():
            for batch in test_data:
                inputs = batch[0].to(self.device)
                outputs = self.model(inputs)

                batch_confidence = F.softmax(outputs, dim=1).max(dim=1)[0].cpu().numpy()
                all_confidences.extend(batch_confidence)

        self.mean_test_confidence = np.mean(all_confidences)

    def evaluate(self, user_data=None, user_confidences=None):
        """
        Evaluate new user data by comparing its average confidence to the test set's mean confidence.

        user_data: DataLoader containing the user dataset for evaluation.
        user_confidences: List of confidence scores for the user data.
        Returns: Boolean indicating if the user data's mean confidence exceeds or matches the test data's mean.
        """
        if user_data is None and user_confidences is None:
            raise ValueError("Either user_data or user_confidences must be provided.")
        elif user_data is not None and user_confidences is not None:
            raise ValueError(
                "Only one of user_data or user_confidences should be provided."
            )

        if user_data is not None:
            self.model.eval()
            all_confidences = []

            with torch.no_grad():
                for batch in user_data:
                    inputs = batch[0].to(self.device)
                    outputs = self.model(inputs)

                    batch_confidence = (
                        F.softmax(outputs, dim=1).max(dim=1)[0].cpu().numpy()
                    )
                    all_confidences.extend(batch_confidence)
        else:
            all_confidences = user_confidences

        # Calculate mean confidence for the user data
        mean_user_confidence = np.mean(all_confidences)

        # Compare mean user confidence to the stored test set mean confidence
        return mean_user_confidence

    def binary_evaluation(self, user_data=None, user_confidences=None):
        """
        Evaluate new user data by comparing its average confidence to the test set's mean confidence.

        user_data: DataLoader containing the user dataset for evaluation.
        user_confidences: List of confidence scores for the user
        Returns: Boolean indicating if the user data's mean confidence exceeds or matches the test data's mean.
        """

        mean_user_confidence = self.evaluate(user_data, user_confidences)

        # Determine if the user data's mean confidence exceeds or matches the test data's mean
        return mean_user_confidence >= self.mean_test_confidence
