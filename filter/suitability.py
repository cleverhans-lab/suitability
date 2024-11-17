import numpy as np
import shap
import torch
import torch.nn.functional as F
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

import suitability.filter.tests as ftests
from sklearn.preprocessing import StandardScaler


class SuitabilityFilter:
    def __init__(
        self, model, test_data, regressor_data, device, use_labels=False, normalize=True
    ):
        """
        model: the model to be evaluated (torch model)
        test_data: the test data used to evaluate the model (torch dataset)
        regressor_data: the sample used by the provider to train the regressor (torch dataset)
        device: the device used to evaluate the model (torch device)
        use_labels: whether to use the GT labels for the regressor (bool)
        normalize: whether to normalize the features (bool)

        """
        self.model = model
        self.test_data = test_data
        self.regressor_data = regressor_data
        self.device = device
        self.use_labels = use_labels

        self.regressor_features = None
        self.regressor_correct = None
        self.regressor = None
        self.scaler = StandardScaler()
        self.normalize = normalize
        self.regressor_subset = {}
    

        self.test_features = None
        self.test_correct = None

    def get_features(self, data):
        """
        Get the features (signals) used by the regressor

        data: the data we want to calculate the features for (torch dataset)

        return: the features (signals) used by the regressor (numpy array), an binary array representing prediction correctness (numpy array)
        """
        self.model.eval()
        all_features = []
        all_correct = []

        with torch.no_grad():
            for batch in data:
                inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
                outputs = self.model(inputs)

                # CORRECTNESS
                predictions = torch.argmax(outputs, dim=1)
                correct = (predictions == labels).cpu().numpy()
                all_correct.extend(correct)

                # SIGNALS
                # Confidence signals
                softmax_outputs = F.softmax(outputs, dim=1)
                conf_mean = softmax_outputs.mean(dim=1).cpu().numpy()
                conf_max = softmax_outputs.max(dim=1)[0].cpu().numpy()
                conf_std = softmax_outputs.std(dim=1).cpu().numpy()
                conf_entropy = (
                    (
                        -torch.sum(
                            softmax_outputs * torch.log(softmax_outputs + 1e-10), dim=1
                        )
                    )
                    .cpu()
                    .numpy()
                )

                # Logits signals
                logit_mean = outputs.mean(dim=1).cpu().numpy()
                logit_max = outputs.max(dim=1)[0].cpu().numpy()
                logit_std = outputs.std(dim=1).cpu().numpy()
                logit_diff_top2 = (
                    (
                        torch.topk(outputs, 2, dim=1).values[:, 0]
                        - torch.topk(outputs, 2, dim=1).values[:, 1]
                    )
                    .cpu()
                    .numpy()
                )

                # Loss signals
                loss = (
                    F.cross_entropy(outputs, predictions, reduction="none")
                    .cpu()
                    .numpy()
                )

                # Margin loss as difference in cross-entropy loss
                top_two_classes = torch.topk(
                    outputs, 2, dim=1
                ).indices  # Get indices of top 2 classes
                pred_class_probs = F.softmax(outputs, dim=1)[
                    range(len(labels)), top_two_classes[:, 0]
                ]
                next_best_class_probs = F.softmax(outputs, dim=1)[
                    range(len(labels)), top_two_classes[:, 1]
                ]

                pred_class_loss = -torch.log(pred_class_probs + 1e-10).cpu().numpy()
                next_best_class_loss = (
                    -torch.log(next_best_class_probs + 1e-10).cpu().numpy()
                )
                margin_loss = (
                    pred_class_loss - next_best_class_loss
                )  # Difference in loss between top 2 classes

                # Gradient Norm (L2 norm of the gradient wrt input)
                # inputs.requires_grad = True
                # loss_grad = F.cross_entropy(outputs, predictions)
                # self.model.zero_grad()
                # loss_grad.backward()
                # grad_norm = inputs.grad.norm(2, dim=(1, 2, 3)).cpu().numpy()

                # Class Probability Ratios
                top_two_probs = (
                    torch.topk(softmax_outputs, 2, dim=1).values.cpu().numpy()
                )
                class_prob_ratio = top_two_probs[:, 0] / (top_two_probs[:, 1] + 1e-10)

                # Top-k Class Probabilities
                top_k = int(
                    softmax_outputs.size(1) * 0.1
                )  # Calculate top 10% class probabilities
                top_k_probs_sum = (
                    torch.topk(softmax_outputs, top_k, dim=1)
                    .values.sum(dim=1)
                    .cpu()
                    .numpy()
                )

                # Energy signals
                energy = -torch.logsumexp(outputs, dim=1).cpu().numpy()

                if self.use_labels:
                    conf_corr = (
                        softmax_outputs[torch.arange(len(labels)), labels].cpu().numpy()
                    )
                    logit_corr = (
                        outputs[torch.arange(len(labels)), labels].cpu().numpy()
                    )
                    loss_corr = (
                        F.cross_entropy(outputs, labels, reduction="none").cpu().numpy()
                    )

                    features = np.column_stack(
                        [
                            # conf_mean,
                            conf_max,
                            conf_std,
                            conf_entropy,
                            logit_mean,
                            logit_max,
                            logit_std,
                            logit_diff_top2,
                            loss,
                            margin_loss,
                            # grad_norm,
                            class_prob_ratio,
                            top_k_probs_sum,
                            conf_corr,
                            logit_corr,
                            loss_corr,
                            energy,
                        ]
                    )
                else:
                    # Combining all signals into a feature vector
                    features = np.column_stack(
                        [
                            # conf_mean,
                            conf_max,
                            conf_std,
                            conf_entropy,
                            logit_mean,
                            logit_max,
                            logit_std,
                            logit_diff_top2,
                            loss,
                            margin_loss,
                            # grad_norm,
                            class_prob_ratio,
                            top_k_probs_sum,
                            energy,
                        ]
                    )

                all_features.append(features)

        features = np.vstack(all_features)
        correct = np.array(all_correct, dtype=bool)

        return features, correct

    def get_correct(self, data):
        """
        Get the correctness of the model predictions

        data: the data we want to calculate the correctness for (torch dataset)

        return: a binary array representing prediction correctness (numpy array)
        """
        self.model.eval()
        all_correct = []

        with torch.no_grad():
            for batch in data:
                inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
                outputs = self.model(inputs)

                predictions = torch.argmax(outputs, dim=1)
                correct = (predictions == labels).cpu().numpy()
                all_correct.extend(correct)

        return np.array(all_correct, dtype=bool)

    def train_regressor(self, calibrated=True):
        """
        Train the regressor using the regressor data

        calibrated: whether the regressor should be calibrated or not (bool)
        """
        if self.regressor_features is None or self.regressor_correct is None:
            features, correct = self.get_features(self.regressor_data)
            self.regressor_features, self.regressor_correct = features, correct
        else:
            features, correct = self.regressor_features, self.regressor_correct

        if self.normalize:
            features = self.scaler.fit_transform(features)

        regression_model = LogisticRegression(max_iter=1000)

        if not calibrated:
            self.regressor = regression_model.fit(features, correct)
        else:
            self.regressor = CalibratedClassifierCV(
                estimator=regression_model, method="sigmoid", cv=5
            ).fit(features, correct)

    def train_regressor_for_feature_subset(self, feature_subset, calibrated=True):
        """
        Train the regressor using the regressor data for a selected subset of signals.

        feature_subset: the subset of features to be used for the regressor (list of integers)
        calibrated: whether the regressor should be calibrated or not (bool)
        """
        if self.regressor_subset.get(tuple(feature_subset), None) is not None:
            return

        if self.regressor_features is None or self.regressor_correct is None:
            features, correct = self.get_features(self.regressor_data)
            self.regressor_features, self.regressor_correct = features, correct
        else:
            features, correct = self.regressor_features, self.regressor_correct

        if self.normalize:
            self.scaler.fit_transform(features)

        features = features[:, feature_subset]

        regression_model = LogisticRegression(max_iter=1000)

        if not calibrated:
            self.regressor_subset[tuple(feature_subset)] = regression_model.fit(
                features, correct
            )
        else:
            self.regressor_subset[tuple(feature_subset)] = CalibratedClassifierCV(
                estimator=regression_model, method="sigmoid", cv=5
            ).fit(features, correct)

    def suitability_test(
        self,
        user_data=None,
        user_features=None,
        margin=0,
        test_power=False,
        get_sample_size=False,
        return_predictions=False,
    ):
        """
        Perform the suitability test

        user_data: the data provided by the user to evaluate suitability (torch dataset)
        user_features: the features (signals) used by the regressor for the user data (numpy array)
        margin: the margin used for the non-inferiority test (float)
        test_power: whether to calculate the test power (bool)
        get_sample_size: whether to calculate the sample size required for the test (bool)
        """
        if self.regressor is None:
            raise ValueError("Regressor not trained")

        if self.test_features is None or self.test_correct is None:
            test_features, test_correct = self.get_features(self.test_data)
            self.test_features, self.test_correct = test_features, test_correct
        else:
            test_features, test_correct = self.test_features, self.test_correct

        if user_data is not None:
            assert (
                user_features is None
            ), "Either user_data or user_features should be None"
            user_features, _ = self.get_features(self.user_data)
        elif user_features is not None:
            assert user_data is None, "Either user_data or user_features should be None"
        else:
            raise ValueError("Either user_data or user_features should be provided")

        if self.normalize:
            test_features = self.scaler.transform(test_features)
            user_features = self.scaler.transform(user_features)

        test_predictions = self.regressor.predict_proba(test_features)[:, 1]
        user_predictions = self.regressor.predict_proba(user_features)[:, 1]

        test = ftests.non_inferiority_ttest(
            test_predictions, user_predictions, margin=margin
        )

        if test_power:
            power = ftests.power_non_inferiority_ttest(
                test_predictions, user_predictions, margin=margin
            )
            test["power"] = power

        if get_sample_size:
            sample_size = ftests.sample_size_non_inferiority_ttest(
                test_predictions, user_predictions, power=0.8, margin=margin
            )
            test["sample_size_0.8_power"] = sample_size

        if return_predictions:
            test["test_predictions"] = test_predictions
            test["user_predictions"] = user_predictions

        return test

    def suitability_test_for_individual_features(
        self,
        user_data=None,
        user_features=None,
        test_power=False,
        get_sample_size=False,
    ):
        """
        Perform the suitability test for each individual feature.

        user_data: the data provided by the user to evaluate suitability (torch dataset)
        user_features: the features (signals) used by the regressor for the user data (numpy array)
        test_power: whether to calculate the test power (bool)
        get_sample_size: whether to calculate the sample size required for the test (bool)
        """

        if self.test_features is None or self.test_correct is None:
            test_features, test_correct = self.get_features(self.test_data)
            self.test_features, self.test_correct = test_features, test_correct
        else:
            test_features, test_correct = self.test_features, self.test_correct

        if user_data is not None:
            assert (
                user_features is None
            ), "Either user_data or user_features should be None"
            user_features, _ = self.get_features(self.user_data)
        elif user_features is not None:
            assert user_data is None, "Either user_data or user_features should be None"
        else:
            raise ValueError("Either user_data or user_features should be provided")

        # Ensure user_features and test_features are in a comparable format (e.g., 2D numpy arrays)
        if user_features.shape[1] != test_features.shape[1]:
            raise ValueError(
                "Mismatch in number of features between user and test data."
            )

        # Array to store test results for each feature
        feature_tests = []

        for i in range(user_features.shape[1]):
            user_feature = user_features[:, i]
            test_feature = test_features[:, i]

            # Perform non-inferiority t-test with margin of 0 for each feature
            test = ftests.non_inferiority_ttest(test_feature, user_feature, margin=0)

            if test_power:
                power = ftests.power_non_inferiority_ttest(
                    test_feature, user_feature, margin=0
                )
                test["power"] = power

            if get_sample_size:
                sample_size = ftests.sample_size_non_inferiority_ttest(
                    test_feature, user_feature, power=0.8, margin=0
                )
                test["sample_size_0.8_power"] = sample_size

            # Append the test dictionary for the current feature
            feature_tests.append(test)

        # Return the array of test dictionaries, one for each feature
        return feature_tests

    def suitability_test_for_feature_subset(
        self,
        feature_subset,
        user_data=None,
        user_features=None,
        margin=0,
        test_power=False,
        get_sample_size=False,
        return_predictions=False,
        calibrated=True,
    ):
        """
        Perform the suitability test for a selected subset of signals.

        feature_subset: the subset of features to be used for the test (list of integers)
        user_data: the data provided by the user to evaluate suitability (torch dataset)
        user_features: the features (signals) used by the regressor for the user data (numpy array)
        margin: the margin used for the non-inferiority test (float)
        test_power: whether to calculate the test power (bool)
        get_sample_size: whether to calculate the sample size required for the test (bool)
        calibrated: whether the regressor should be calibrated or not (bool)
        """
        if self.regressor_subset.get(tuple(feature_subset), None) is None:
            self.train_regressor_for_feature_subset(
                feature_subset=feature_subset, calibrated=calibrated
            )

        if self.test_features is None or self.test_correct is None:
            test_features, test_correct = self.get_features(self.test_data)
            self.test_features, self.test_correct = test_features, test_correct
        else:
            test_features, test_correct = self.test_features, self.test_correct

        if user_data is not None:
            assert (
                user_features is None
            ), "Either user_data or user_features should be None"
            user_features, _ = self.get_features(self.user_data)
        elif user_features is not None:
            assert user_data is None, "Either user_data or user_features should be None"
        else:
            raise ValueError("Either user_data or user_features should be provided")

        if self.normalize:
            test_features = self.scaler.transform(test_features)
            user_features = self.scaler.transform(user_features)

        user_features = user_features[:, feature_subset]
        test_features = test_features[:, feature_subset]

        test_predictions = self.regressor_subset[tuple(feature_subset)].predict_proba(
            test_features
        )[:, 1]
        user_predictions = self.regressor_subset[tuple(feature_subset)].predict_proba(
            user_features
        )[:, 1]

        test = ftests.non_inferiority_ttest(
            test_predictions, user_predictions, margin=margin
        )

        if test_power:
            power = ftests.power_non_inferiority_ttest(
                test_predictions, user_predictions, margin=margin
            )
            test["power"] = power

        if get_sample_size:
            sample_size = ftests.sample_size_non_inferiority_ttest(
                test_predictions, user_predictions, power=0.8, margin=margin
            )
            test["sample_size_0.8_power"] = sample_size

        if return_predictions:
            test["test_predictions"] = test_predictions
            test["user_predictions"] = user_predictions

        return test

    def performance_equivalence_test(
        self,
        user_data_1=None,
        user_features_1=None,
        user_data_2=None,
        user_features_2=None,
        margin=0,
    ):
        """
        Perform the performance equivalence test between two user datasets

        user_data_1: the first data provided by the user to evaluate performance (torch dataset)
        user_features_1: the features (signals) used by the regressor for the first user data (numpy array)
        user_data_2: the second data provided by the user to evaluate performance (torch dataset)
        user_features_2: the features (signals) used by the regressor for the second user data (numpy array)
        margin: the margin used for the equivalence test bounds (float)
        """
        if self.regressor is None:
            raise ValueError("Regressor not trained")

        if user_data_1 is not None:
            assert (
                user_features_1 is None
            ), "Either user_data_1 or user_features_1 should be None"
            user_features_1, _ = self.get_features(user_data_1)
        elif user_features_1 is not None:
            assert (
                user_data_1 is None
            ), "Either user_data_1 or user_features_1 should be None"
        else:
            raise ValueError("Either user_data_1 or user_features_1 should be provided")

        if user_data_2 is not None:
            assert (
                user_features_2 is None
            ), "Either user_data_2 or user_features_2 should be None"
            user_features_2, _ = self.get_features(user_data_2)
        elif user_features_2 is not None:
            assert (
                user_data_2 is None
            ), "Either user_data_2 or user_features_2 should be None"
        else:
            raise ValueError("Either user_data_2 or user_features_2 should be provided")

        user_predictions_1 = self.regressor.predict_proba(user_features_1)[:, 1]
        user_predictions_2 = self.regressor.predict_proba(user_features_2)[:, 1]

        test = ftests.equivalence_test(
            user_predictions_1,
            user_predictions_2,
            threshold_low=-margin,
            threshold_upp=margin,
        )

        return test

    def suitability_test_with_correctness(
        self, user_data=None, margin=0, test_power=False, get_sample_size=False
    ):
        """
        Perform the suitability test

        user_data: the data provided by the user to evaluate suitability (torch dataset)
        margin: the margin used for the non-inferiority test (float)
        test_power: whether to calculate the test power (bool)
        get_sample_size: whether to calculate the sample size required for the test (bool)
        """

        test_conf = self.get_correct(self.test_data)
        user_conf = self.get_correct(user_data)

        test = ftests.non_inferiority_ttest(test_conf, user_conf, margin=margin)

        if test_power:
            power = ftests.power_non_inferiority_ttest(
                test_conf, user_conf, margin=margin
            )
            test["power"] = power

        if get_sample_size:
            sample_size = ftests.sample_size_non_inferiority_ttest(
                test_conf, user_conf, power=0.8, margin=margin
            )
            test["sample_size_0.8_power"] = sample_size

        return test

    def calculate_shap_values(self, sample_data=None, K=100):
        """
        Calculate SHAP values for the trained regressor.

        sample_data: Optional. A subset of data for SHAP value calculation
                    (to save computation time).
        K: Optional. The number of samples to use for summarizing the background data
        if the dataset is large (default is 100).

        return: SHAP values as a numpy array.
        """
        if self.regressor is None:
            raise ValueError(
                "The regressor is not trained. Train it using `train_regressor` before calling this function."
            )

        # Use either provided sample data or all features
        data_to_explain = (
            sample_data if sample_data is not None else self.regressor_features
        )

        # Create a background dataset to use for SHAP value calculations
        if (
            len(self.regressor_features) > 1000
        ):  # Arbitrary threshold for large datasets
            # Summarize the background data
            summarized_background = shap.kmeans(self.regressor_features, K)
        else:
            summarized_background = self.regressor_features

        # Initialize SHAP KernelExplainer with summarized background data
        explainer = shap.KernelExplainer(
            self.regressor.predict_proba, summarized_background
        )

        # Calculate SHAP values for each feature
        shap_values = explainer.shap_values(data_to_explain)

        return shap_values, explainer
