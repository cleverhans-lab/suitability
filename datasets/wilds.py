from examples.transforms import initialize_transform
from torch.utils.data import DataLoader, Dataset

from wilds import get_dataset

"""
WILDS is a benchmark of in-the-wild distribution shifts spanning diverse data modalities and applications, from tumor identification to wildlife monitoring to poverty mapping.
More information on the benchmark and datasets can be found at https://wilds.stanford.edu/.
"""


class ResConfig:
    def __init__(self, dataset_name):
        self.target_resolution = self.get_target_resolution(dataset_name)

    def get_target_resolution(self, dataset_name):
        if dataset_name == "iwildcam":
            return (448, 448)
        elif dataset_name == "fmow":
            return (224, 224)
        elif dataset_name == "rxrx1":
            return (256, 256)
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")


class BertConfig:
    def __init__(self, dataset_name):
        if dataset_name == "civilcomments":
            self.model = "distilbert-base-uncased"
            self.max_token_length = 300
        elif dataset_name == "amazon":
            self.model = "distilbert-base-uncased"
            self.max_token_length = 512
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")
        self.pretrained_model_path = None
        self.model_kwargs = {}


class WILDSDataset(Dataset):
    def __init__(self, dataset_name, split, root_dir, pre_filter={}):
        """
        dataset_name: the name of the dataset (string)
        split: the split of the dataset (string)
        root_dir: the root directory of the dataset (string)
        pre_filter: the filter to be applied to the data (dict)
        transform: the transformation to be applied to the data (torchvision transform)
        """
        dataset = get_dataset(dataset=dataset_name, download=False, root_dir=root_dir)

        if dataset_name in ["civilcomments", "amazon"]:
            config = BertConfig(dataset_name)
            transform = initialize_transform(
                transform_name="bert", config=config, dataset=dataset, is_training=False
            )
        elif dataset_name in ["iwildcam", "fmow"]:
            config = ResConfig(dataset_name)
            transform = initialize_transform(
                transform_name="image_base",
                config=config,
                dataset=dataset,
                is_training=False,
            )
        elif dataset_name == "rxrx1":
            config = ResConfig(dataset_name)
            transform = initialize_transform(
                transform_name="rxrx1",
                config=config,
                dataset=dataset,
                is_training=False,
            )

        self.dataset = dataset.get_subset(split, transform=transform)
        self.filtered_indices = list(range(len(self.dataset)))

        for key, value in pre_filter.items():
            # Filter the fmow dataset based on pre-filter
            if dataset_name == "fmow":
                if key == "region":
                    key_ind = 0
                    if value == "Asia":
                        val_ind = 0
                    elif value == "Europe":
                        val_ind = 1
                    elif value == "Africa":
                        val_ind = 2
                    elif value == "Americas":
                        val_ind = 3
                    elif value == "Oceania":
                        val_ind = 4
                    else:
                        raise ValueError(f"Region {value} not supported")
                elif key == "year":
                    key_ind = 1
                    assert (
                        value >= 2002 and value <= 2017
                    ), "Year must be between 2002 and 2017"
                    val_ind = value - 2002
                else:
                    raise ValueError(f"Filter property {key} not supported")
                self.filtered_indices = [
                    i
                    for i in self.filtered_indices
                    if self.dataset[i][2][key_ind] == val_ind
                ]

            if dataset_name == "civilcomments":
                if key == "sensitive":
                    if value == "male":
                        val_ind = 0
                    elif value == "female":
                        val_ind = 1
                    elif value == "LGBTQ":
                        val_ind = 2
                    elif value == "christian":
                        val_ind = 3
                    elif value == "muslim":
                        val_ind = 4
                    elif value == "other_religions":
                        val_ind = 5
                    elif value == "black":
                        val_ind = 6
                    elif value == "white":
                        val_ind = 7
                else:
                    raise ValueError(f"Filter property {key} not supported")
                self.filtered_indices = [
                    i for i in self.filtered_indices if self.dataset[i][2][val_ind] == 1
                ]

            if dataset_name == "rxrx1":
                if key == "cell_type":
                    key_ind = 0
                    if value == "HEPG2":
                        val_ind = 0
                    elif value == "HUVEC":
                        val_ind = 1
                    elif value == "RPE":
                        val_ind = 2
                    elif value == "U2OS":
                        val_ind = 3
                    else:
                        raise ValueError(f"Cell type {value} not supported")
                else:
                    raise ValueError(f"Filter property {key} not supported")

                self.filtered_indices = [
                    i
                    for i in self.filtered_indices
                    if self.dataset[i][2][key_ind] == val_ind
                ]

            else:
                raise ValueError(f"Filtering not supported for dataset {dataset_name}")

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        return self.dataset[self.filtered_indices[idx]]


def get_wilds_dataset(
    dataset_name, root_dir, split, batch_size, shuffle, num_workers, pre_filter={}
):
    assert dataset_name in [
        "iwildcam",
        "fmow",
        "civilcomments",
        "rxrx1",
        "amazon",
    ], f"Dataset {dataset_name} not supported"
    assert split in [
        "val",
        "test",
        "id_val",
        "id_test",
    ], "Split must be (id_)val or (id_)test, this should be used for evaluation only"
    dataset = WILDSDataset(dataset_name, split, root_dir, pre_filter=pre_filter)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return dataloader
