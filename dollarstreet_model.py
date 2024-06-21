import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import wandb
import os
import argparse
from datasets.dollarstreet import get_dollarstreet

class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value_str = value.split('=')
            if value_str.replace('-','').isnumeric():
                processed_val = int(value_str)
            elif value_str.replace('-','').replace('.','').isnumeric():
                processed_val = float(value_str)
            elif value_str in ['True', 'true']:
                processed_val = True
            elif value_str in ['False', 'false']:
                processed_val = False
            else:
                processed_val = value_str
            getattr(namespace, self.dest)[key] = processed_val

# Function to read wandb API key from a file and login
def login_to_wandb(wandb_api_key_path):
    with open(wandb_api_key_path, 'r') as file:
        api_key = file.read().strip()
    wandb.login(key=api_key)

# Function to train the model
def train_model(train_loader, test_loader, model, criterion, optimizer, scheduler, num_epochs, checkpoint_dir, device):
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        # Training loop
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        # Log the train loss and accuracy
        wandb.log({'epoch': epoch, 'train_loss': epoch_loss, 'train_accuracy': epoch_acc})

        # Evaluate on the test set
        model.eval()
        test_loss = 0.0
        test_corrects = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                test_loss += loss.item() * inputs.size(0)
                test_corrects += torch.sum(preds == labels.data)

        test_loss = test_loss / len(test_loader.dataset)
        test_acc = test_corrects.double() / len(test_loader.dataset)

        # Log the test loss and accuracy
        wandb.log({'epoch': epoch, 'test_loss': test_loss, 'test_accuracy': test_acc})

        # Checkpointing
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)

        last_model_path = os.path.join(checkpoint_dir, 'last_model.pth')
        torch.save(model.state_dict(), last_model_path)

        # Step the scheduler
        scheduler.step()

# Define the main function
def main(args):
    # Login to wandb
    login_to_wandb(args.wandb_api_key_path)

    # Initialize wandb
    wandb.init(project=args.wandb_kwargs['project'])

    # Log config parameters to wandb
    wandb.config.update(args)

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the ResNet50 model
    model = models.resnet50(pretrained=False)
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # Replace these with code to initialize your DataLoader
    train_loader = get_dollarstreet(args.root_dir, "train", args.batch_size, shuffle=True, num_workers=4)
    test_loader = get_dollarstreet(args.root_dir, "test", args.batch_size, shuffle=False, num_workers=4)

    train_model(train_loader, test_loader, model, criterion, optimizer, scheduler, args.epochs, args.log_dir, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet50 on DollarStreet and log metrics to wandb")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory for the dataset")
    parser.add_argument("--log_dir", type=str, required=True, help="Directory to store logs and checkpoints")
    parser.add_argument("--use_wandb", action="store_true", help="Flag to use Weights & Biases for logging")
    parser.add_argument("--wandb_api_key_path", type=str, required=True, help="Path to the wandb API key file")
    parser.add_argument("--wandb_kwargs", nargs='*', action=ParseKwargs, default={}, help='keyword arguments for wandb.init() passed as key1=value1 key2=value2')
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=90, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for the optimizer")
    parser.add_argument("--lr_step_size", type=int, default=30, help="Step size for the learning rate scheduler")
    parser.add_argument("--lr_gamma", type=float, default=0.1, help="Gamma for the learning rate scheduler")

    args = parser.parse_args()
    main(args)
