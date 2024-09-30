from suitability.filter.sample_signals import (
    CorrectnessSignal,
    ConfidenceSignal,
    LogitSignal,
    DecisionBoundarySignal,
    TrainingDynamicsSignal_Basic,
    TrainingDynamicsSignal,
)
import suitability.filter.tests as ftests
from suitability.filter.margin_tuning import tune_margin
import pandas as pd
import numpy as np


class SuitabilityFilter:
    def __init__(
        self,
        model,
        eval_data,
        tuning_data,
        test_sample,
        device,
        signals,
        target_accuracy,
    ):
        """
        model: the model to be evaluated (torch model)
        eval_data: the data used to evaluate the model (torch dataloader)
        tuning_data: the data used to tune the margin (torch dataloader)
        test_sample: the data provided by the user (torch dataloader)
        signals: the signals used to evaluate the model (string array)

        """
        self.model = model
        self.eval_data = eval_data
        self.tuning_data = tuning_data
        self.test_sample = test_sample
        self.device = device
        self.signals = signals
        self.target_accuracy = target_accuracy

        columns = {
            "signal": pd.Series(
                dtype="str"
            ),  # 'str' for the signal column (string type)
            "eval_data": pd.Series(dtype=np.dtype("O")),  # 'O' for object (np arrays)
            "tuning_data": pd.Series(dtype=np.dtype("O")),  # 'O' for object (np arrays)
            "test_sample": pd.Series(dtype=np.dtype("O")),  # 'O' for object (np arrays)
            "margin": pd.Series(dtype="float"),
            "p_value": pd.Series(dtype="float"),
            "test_statistic": pd.Series(dtype="float"),
        }

        self.signal_df = pd.DataFrame(columns)

    def evaluate_signal(self, signal_name, signal_config):
        """
        Compute and evaluate a single signal for the given signal configuration

        signal_name: the signal to be evaluated
        signal_config: a dictionary containing the configuration for each signal
        """
        if signal_name == "correctness":
            signal = CorrectnessSignal(self.model, self.device)
        elif signal_name == "confidence":
            signal = ConfidenceSignal(self.model, self.device)
        elif signal_name == "logit":
            signal = LogitSignal(self.model, self.device)
        elif signal_name == "decision_boundary":
            signal = DecisionBoundarySignal(self.model, self.device)
        elif signal_name == "training_dynamics_basic":
            signal = TrainingDynamicsSignal_Basic(self.model, self.device)
        elif signal_name == "training_dynamics":
            signal = TrainingDynamicsSignal(self.model, self.device)
        else:
            raise ValueError(f"Invalid signal name {signal_name}")

        cs = CorrectnessSignal(self.model, self.device)
        tuning_data_correctness = cs.evaluate(self.tuning_data)

        eval_data_signal = signal.evaluate(self.eval_data, **signal_config)
        tuning_data_signal = signal.evaluate(self.tuning_data, **signal_config)
        test_sample_signal = signal.evaluate(self.test_sample, **signal_config)

        margin = tune_margin(
            correctness = tuning_data_correctness,
            signals = tuning_data_signal,
            bm_signals = eval_data_signal,
            test_fn = ftests.non_inferiority_ttest,
            x = self.target_accuracy,
            **signal_config,
        )

        stat_test = ftests.non_inferiority_ttest(
            sample1 = eval_data_signal,
            sample2 = test_sample_signal,
            margin = margin,
            **signal_config,
        )

        df_row = {
            "signal": signal_name,
            "eval_data": eval_data_signal,
            "tuning_data": tuning_data_signal,
            "test_sample": test_sample_signal,
            "margin": margin,
            "p_value": stat_test["p_value"],
            "test_statistic": stat_test["t_statistic"],
        }

        self.signal_df = pd.concat([self.signal_df, pd.DataFrame(df_row)], ignore_index=True)

    