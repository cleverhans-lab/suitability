import numpy as np
import torch
import torch.nn.functional as F


class AverageThresholdConfidence:
    def __init__(self, model, test_data, device):
        """
        Initialize ATC with a model, reference data, and device.
        Calculates accuracy on reference data as the performance metric value.

        model: PyTorch model with binary classifier (outputting raw scores).
        test_data: DataLoader with reference data for performance metric calculation.
        device: Device to run computations on (e.g., 'cpu' or 'cuda').
        """
        self.model = model
        self.test_data = test_data
        self.device = device

        self.model.eval()

        # List to store confidence scores and correct labels for accuracy calculation
        all_confidences = []
        all_correct = []

        with torch.no_grad():
            for batch in test_data:
                inputs, labels = batch[0].to(self.device), batch[1].to(self.device)

                # Forward pass
                outputs = self.model(inputs)

                # Store correctness for accuracy calculation
                predictions = torch.argmax(outputs, dim=1)
                correct = (predictions == labels).cpu().numpy()
                all_correct.extend(correct)

                # Calculate confidence scores using maximum confidence
                max_confidences = F.softmax(outputs, dim=1).max(dim=1)[0].cpu().numpy()
                all_confidences.extend(max_confidences)

        # Calculate model accuracy on reference data as the performance metric
        self.test_accuracy = np.mean(all_correct)

        # Determine confidence threshold based on the accuracy metric
        sorted_confidences = np.sort(all_confidences)
        threshold_index = int(len(sorted_confidences) * (1 - self.test_accuracy))
        self.confidence_threshold = sorted_confidences[threshold_index]

    def evaluate(self, user_data=None, user_confidences=None):
        """
        Evaluate the model on new data based on the learned confidence threshold.

        user_data: DataLoader with user data for evaluation.
        user_confidences: List of confidence scores for user data.
        Returns: Fraction of observations with confidence above threshold.
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

                    # Apply maximum confidence function MC
                    batch_confidence = (
                        F.softmax(outputs, dim=1).max(dim=1)[0].cpu().numpy()
                    )
                    all_confidences.extend(batch_confidence)
        else:
            all_confidences = user_confidences

        # Calculate fraction of observations above the learned threshold
        above_threshold_fraction = np.mean(
            np.array(all_confidences) >= self.confidence_threshold
        )

        return above_threshold_fraction

    def binary_evaluation(self, user_data=None, user_confidences=None):
        """
        Evaluate new user data by comparing its average confidence to the test set's mean confidence.

        user_data: DataLoader containing the user dataset for evaluation.
        user_confidences: List of confidence scores for user data.
        Returns: Boolean indicating if the user data's mean confidence exceeds or matches the test data's mean.
        """
        above_threshold_fraction = self.evaluate(user_data, user_confidences)

        return above_threshold_fraction >= self.test_accuracy
