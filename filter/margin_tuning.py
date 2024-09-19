import random

import torch


def get_subsample_with_accuracy(correctness, target_accuracy, subsample_size):
    """
    Returns a binary mask indicating the indices included in the subsample.

    Parameters:
    correctness (torch.Tensor or list): A binary array (1s and 0s) indicating if samples are correctly classified.
    target_accuracy (float): The target accuracy for the subsample.
    subsample_size (int): The size of the subsample.

    Returns:
    subsample (torch.Tensor): A binary mask (1s for indices included in the subsample, 0s otherwise).
    actual_accuracy (float): The actual accuracy of the subsample.
    """
    total_samples = len(correctness)
    assert 0 <= target_accuracy <= 1, "Target accuracy must be between 0 and 1."
    assert (
        subsample_size <= total_samples
    ), "Subsample size cannot be larger than the dataset."

    # Separate indices for correctly and incorrectly classified samples
    correct_indices = [i for i, c in enumerate(correctness) if c == 1]
    incorrect_indices = [i for i, c in enumerate(correctness) if c == 0]

    # Number of correctly classified samples needed in the subsample to meet the target accuracy
    num_correct_needed = int(target_accuracy * subsample_size)
    num_incorrect_needed = subsample_size - num_correct_needed

    # Randomly select the required number of correct and incorrect samples
    selected_correct = random.sample(
        correct_indices, min(num_correct_needed, len(correct_indices))
    )
    selected_incorrect = random.sample(
        incorrect_indices, min(num_incorrect_needed, len(incorrect_indices))
    )

    # Combine the selected indices and create a binary mask
    selected_indices = selected_correct + selected_incorrect
    subsample = torch.zeros(total_samples, dtype=torch.int)
    subsample[selected_indices] = 1

    # Calculate the actual accuracy of the subsample
    actual_accuracy = len(selected_correct) / subsample_size

    return subsample, actual_accuracy


def tune_margin(
    correctness,
    signals,
    bm_signals,
    test_fn,
    x,
    delta_x=0.1,
    initial_m=0.05,
    delta_m=0.005,
    accuracy_threshold=0.05,
    max_iters=100,
    subsample_size=0,
):
    """
    Tune the margin parameter m for the non-inferiority test.

    Parameters:
    correctness (torch.Tensor or list): A binary array (1s and 0s) indicating if samples are correctly classified.
    signals (torch.Tensor): The signals to be tested.
    bm_signals (torch.Tensor): The benchmark signals used for the test.
    x (float): The target accuracy for the subsample.
    delta_x (float): The maximum deviation from the target accuracy.
    initial_m (float): The initial margin parameter m.
    delta_m (float): The step size for updating the margin parameter.
    accuracy_threshold (float): The threshold for the subsample accuracy.
    max_iters (int): The maximum number of iterations to run.

    Returns:
    float: The tuned margin parameter m.
    """
    m = initial_m
    if subsample_size == 0:
        subsample_size = len(bm_signals)

    is_increased, is_decreased = 0, 0

    for iteration in range(max_iters):
        # Take several subsamples and check p-values
        for _ in range(5):  # Taking 5 subsamples in each iteration
            target_acc = x + random.uniform(-delta_x, delta_x)
            subsample, accuracy = get_subsample_with_accuracy(
                correctness, target_accuracy=target_acc, subsample_size=subsample_size
            )

            p_value = test_fn(bm_signals, signals[subsample == 1], m)["p_value"]

            if (
                accuracy < x and p_value <= 0.05
            ):  # Accuracy is below threshold, but p-value is also below 0.05
                print(
                    f"Iteration {iteration}: Accuracy {accuracy:.3f} < {x}, but p-value {p_value:.3f} <= 0.05. Decreasing m to {m-delta_m:.3f}."
                )
                m -= delta_m  # Decrease m to make the test stricter
                is_decreased += 1
                break

            elif (
                accuracy >= x and p_value > 0.05
            ):  # Accuracy is greater than threshold, but p-value is also above 0.05
                print(
                    f"Iteration {iteration}: Accuracy {accuracy:.3f} >= {x}, but p-value {p_value:.3f} > 0.05. Increasing m to {m+delta_m:.3f}."
                )
                m += delta_m  # Increase m to make the test more lenient
                is_increased += 1
                break

    print(
        f"Convergence sanity check: Margin has been increased {is_increased} times and decreased {is_decreased} times out of {max_iters} iterations."
    )

    return m
