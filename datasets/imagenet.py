import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.models import ResNet50_Weights, resnet50


def get_imagenet_dataset(root_dir, splits, batch_size=64, num_workers=4):
    transform = transforms.Compose(
        [
            transforms.Resize(
                size=(232, 232), interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.CenterCrop(size=(224, 224)),
            transforms.ToTensor(),  # Convert to Tensor and scale to [0, 1]
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # Normalize
        ]
    )

    imagenet_data = torchvision.datasets.ImageNet(
        root_dir, split="val", transform=transform
    )

    test, regressor, user = random_split(imagenet_data, splits)
    test_data = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=4)
    regressor_data = DataLoader(regressor, batch_size=64, shuffle=True, num_workers=4)
    user_data = DataLoader(user, batch_size=64, shuffle=False, num_workers=4)

    return test_data, regressor_data, user_data


def get_imagenet_model(model_name):
    if model_name == "resnet50":
        return resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    else:
        raise ValueError(f"Model {model_name} not supported")
