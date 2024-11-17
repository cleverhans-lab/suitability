import numpy as np

def split_dataset_into_folds(correctness, target_accuracies):
    # Define target accuracies
    n_subsets = len(target_accuracies)
    total_samples = len(correctness)

    # Shuffle the indices and the correctness array
    indices = np.random.permutation(total_samples)
    shuffled_correctness = correctness[indices]

    # Initialize subsets
    folds = []
    subsets_correctness = []

    start_idx = 0
    for i, target_accuracy in enumerate(target_accuracies):
        # Calculate the number of correct and incorrect samples needed for the subset
        subset_size = total_samples // n_subsets
        n_correct = int(subset_size * target_accuracy)
        n_incorrect = subset_size - n_correct

        # Find the correct/incorrect samples in the shuffled data
        remaining_correct_indices = np.where(shuffled_correctness[start_idx:] == 1)[0]
        remaining_incorrect_indices = np.where(shuffled_correctness[start_idx:] == 0)[0]

        # Adjust n_correct and n_incorrect based on available samples
        n_correct = min(n_correct, len(remaining_correct_indices))
        n_incorrect = min(n_incorrect, len(remaining_incorrect_indices))

        # Get the actual indices for the current subset
        correct_indices = remaining_correct_indices[:n_correct]
        incorrect_indices = remaining_incorrect_indices[:n_incorrect]

        # Get the final indices for the current subset
        subset_indices = (
            np.concatenate((correct_indices, incorrect_indices)) + start_idx
        )

        # Extract the correctness for the current subset
        folds.append(indices[subset_indices])
        subsets_correctness.append(np.mean(shuffled_correctness[subset_indices]))

        # Move the start index forward
        start_idx += len(subset_indices)

    return folds, subsets_correctness