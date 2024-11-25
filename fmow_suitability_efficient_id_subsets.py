import os
import pickle
import random
from itertools import chain, combinations

import numpy as np
import pandas as pd
import torch

from datasets.wilds import get_wilds_dataset, get_wilds_model
from filter.suitability_efficient import SuitabilityFilter, get_sf_features

# Set seeds for reproducibility
random.seed(32)
np.random.seed(32)

# Define splits for evaluation
valid_id_splits = [
    ("id_val", {"year": [2002, 2003, 2004, 2005, 2006]}),
    ("id_val", {"year": [2007, 2008, 2009]}),
    ("id_val", {"year": [2010]}),
    ("id_val", {"year": [2011]}),
    ("id_val", {"year": [2012]}),
    ("id_val", {"region": ["Asia"]}),
    ("id_val", {"region": ["Europe"]}),
    ("id_val", {"region": ["Americas"]}),
    ("id_test", {"year": [2002, 2003, 2004, 2005, 2006]}),
    ("id_test", {"year": [2007, 2008, 2009]}),
    ("id_test", {"year": [2010]}),
    ("id_test", {"year": [2011]}),
    ("id_test", {"year": [2012]}),
    ("id_test", {"region": ["Asia"]}),
    ("id_test", {"region": ["Europe"]}),
    ("id_test", {"region": ["Americas"]}),
]

# Configuration
data_name = "fmow"
root_dir = "/mfsnic/projects/suitability/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
algorithm = "ERM"
model_type = "last"
seed = 0
model = get_wilds_model(
    data_name, root_dir, algorithm=algorithm, seed=seed, model_type=model_type
)
model = model.to(device)
model.eval()
print(f"Model loaded to device: {device}")

margins = [0]

# Initialize results DataFrame
sf_results = []
cache_file = "suitability/results/features/fmow_id.pkl"

if os.path.exists(cache_file):
    with open(cache_file, "rb") as f:
        id_splits_features_cache = pickle.load(f)
        print("Features loaded")
else:
    id_splits_features_cache = {}

    # Precompute all data features
    for split_name, split_filter in valid_id_splits:
        print(f"Computing features for split: {split_name} with filter {split_filter}")
        dataset = get_wilds_dataset(
            data_name,
            root_dir,
            split_name,
            batch_size=64,
            shuffle=False,
            num_workers=4,
            pre_filter=split_filter,
        )
        id_splits_features_cache[(split_name, str(split_filter))] = get_sf_features(
            dataset, model, device
        )
    print("ID splits features computed")

    # Save cache
    with open(cache_file, "wb") as f:
        pickle.dump(id_splits_features_cache, f)
    print("Features saved")

# Combine all features and correctness
all_features = []
all_corr = []
for (split_name, split_filter), (features, corr) in id_splits_features_cache.items():
    all_features.append(features)
    all_corr.append(corr)
all_features = np.concatenate(all_features, axis=0)
all_corr = np.concatenate(all_corr, axis=0)

# Generate indices for all data
indices = np.arange(len(all_corr))
num_folds = 15  # Split remaining data into 15 folds

offset = 0

classifiers = [
    "logistic_regression"
]  # "logistic_regression", "svm", "random_forest", "gradient_boosting", "mlp", "decision_tree"]
margins = [0]
normalize = True
calibrated = True

elements = list(range(0, 12))


def all_non_empty_subsets(iterable):
    return list(
        chain.from_iterable(
            combinations(iterable, r) for r in range(1, len(iterable) + 1)
        )
    )


feature_subsets = [list(subset) for subset in all_non_empty_subsets(elements)]

# Main loop
for user_split_name, user_filter in valid_id_splits:
    print(f"Evaluating user split: {user_split_name} with filter {user_filter}")
    user_features, user_corr = id_splits_features_cache[
        (user_split_name, str(user_filter))
    ]
    user_size = len(user_corr)
    user_acc = np.mean(user_corr)

    # Identify indices to exclude for user_split
    user_indices = np.arange(offset, offset + user_size)
    offset += user_size

    # Create new indices excluding user_indices
    remaining_indices = np.setdiff1d(indices, user_indices)
    np.random.shuffle(remaining_indices)

    # Re-partition remaining data into folds
    num_remaining_samples = len(remaining_indices)
    remaining_fold_size = num_remaining_samples // num_folds
    fold_indices = [
        remaining_indices[i * remaining_fold_size : (i + 1) * remaining_fold_size]
        for i in range(num_folds)
    ]

    for i, reg_indices in enumerate(fold_indices):
        reg_features = all_features[reg_indices]
        reg_corr = all_corr[reg_indices]
        reg_size = len(reg_corr)
        reg_acc = np.mean(reg_corr)

        for j, test_indices in enumerate(fold_indices):
            if i == j:
                continue
            test_features = all_features[test_indices]
            test_corr = all_corr[test_indices]
            test_size = len(test_corr)
            test_acc = np.mean(test_corr)

            for classifier in classifiers:
                for feature_subset in feature_subsets:
                    suitability_filter = SuitabilityFilter(
                        model,
                        test_features,
                        test_corr,
                        reg_features,
                        reg_corr,
                        device,
                        normalize=normalize,
                        feature_subset=feature_subset,
                    )
                    suitability_filter.train_classifier(
                        calibrated=calibrated, classifier=classifier
                    )

                    for margin in margins:
                        sf_test = suitability_filter.suitability_test(
                            user_features=user_features, margin=margin
                        )
                        p_value = sf_test["p_value"]
                        ground_truth = user_acc >= test_acc - margin

                        sf_results.append(
                            {
                                "data_name": data_name,
                                "algorithm": algorithm,
                                "seed": seed,
                                "model_type": model_type,
                                "normalize": normalize,
                                "calibrated": calibrated,
                                "margin": margin,
                                "reg_fold": i,
                                "reg_size": reg_size,
                                "reg_acc": reg_acc,
                                "test_fold": j,
                                "test_size": test_size,
                                "test_acc": test_acc,
                                "user_split": user_split_name,
                                "user_filter": user_filter,
                                "user_size": user_size,
                                "user_acc": user_acc,
                                "p_value": p_value,
                                "ground_truth": ground_truth,
                                "classifier": classifier,
                                "feature_subset": feature_subset,
                            }
                        )

# Save results
sf_evals = pd.DataFrame(sf_results)
sf_evals.to_csv(
    "suitability/results/sf_evals/erm/fmow_sf_results_id_subsets.csv", index=False
)
