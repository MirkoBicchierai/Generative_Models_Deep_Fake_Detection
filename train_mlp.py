import comet_ml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Model import ClassificationMLP
from DataLoader import FlattenedMLPDataset, FastDataset
from tqdm import tqdm
from torch import optim
import numpy as np
import argparse


comet_ml.login(api_key="S8bPmX5TXBAi6879L55Qp3eWW")


lr = 0.0001
batch_train = 64
val_test_batch_size = 128
num_workers = 12

exp = comet_ml.Experiment(
    project_name="Generative Models Project Work",
    auto_metric_logging=False,
    auto_param_logging=False,
)


def train(model, train_dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for inputs, labels in tqdm(train_dataloader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_dataloader)
    return avg_loss


def evaluate_with_voting(model, dataloader, criterion, device):
    model.eval()
    correct_sequences = 0  # For tracking the number of correctly predicted sequences
    total_sequences = 0  # Total number of sequences
    total_loss = 0.0  # Accumulating loss

    with torch.no_grad():
        for sequences, labels in dataloader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            for i in range(len(sequences)):
                sequence = sequences[i]
                label = labels[i]  # Label for the entire sequence
                frame_preds = []
                sequence_loss = 0.0  # Initialize sequence-specific loss

                for frame in sequence:
                    frame = frame.unsqueeze(0).to(device)  # Add batch dimension
                    output = model(frame)  # Get model output

                    # Ensure the label is reshaped for the loss function
                    label_tensor = torch.tensor([label]).to(device)  # Shape: [1]

                    loss = criterion(
                        output, label_tensor
                    )  # Use the correct label for the frame
                    sequence_loss += loss.item()  # Accumulate loss for the sequence

                    _, predicted = torch.max(
                        output, 1
                    )  # Get predicted class for the frame
                    frame_preds.append(predicted.item())

                # Majority vote on frame predictions
                correct_frames = sum(
                    [1 for pred in frame_preds if pred == label.item()]
                )

                # If at least half of the frames are correctly classified, consider the sequence as correct
                if correct_frames >= len(frame_preds) // 2:
                    correct_sequences += 1  # Sequence is considered correct
                total_sequences += 1

                total_loss += sequence_loss
    accuracy = correct_sequences / total_sequences
    average_loss = total_loss / total_sequences

    return average_loss, accuracy


# Function to evaluate the model
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Compute accuracy
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / len(dataloader)
    accuracy = correct / total * 100

    return avg_loss, accuracy


def _parse_args():
    """Parse cmd line args for training an image classifier."""
    parser = argparse.ArgumentParser(description="Train an image classifier")
    parser.add_argument(
        "--one_vs_rest",
        action="store_true",
        help="If true, train a single class vs all other classes",
    )

    parser.add_argument(
        "--classes",
        nargs="+",
        default=["ORIGINAL", "NT"],
        help="List of classes to train on",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="CLIP",
        help="Dataset to use for training",
    )
    return parser.parse_args()


# Main function
def main():

    args = _parse_args()
    one_vs_rest = args.one_vs_rest
    classes = args.classes
    dataset = args.dataset
    if one_vs_rest:
        str_classes = "-".join(classes[1:])
        str_classes = "ORIGINAL vs " + str_classes
    else:
        str_classes = "-".join(classes)

    parameters = {
        "batch_size": batch_train,
        "learning_rate": lr,
        "type": str_classes,
    }
    if one_vs_rest and len(classes) == 6:
        exp_class = "Original vs All"
    elif len(classes) == 6:
        exp_class = "All Classes"
    else:
        exp_class = classes[1]

    exp.set_name(dataset + " - " + "MLP" + " - " + exp_class)

    train_dataset_path = f"Dataset/FF++/{dataset}/train"
    val_dataset_path = f"Dataset/FF++/{dataset}/val"
    test_dataset_path = f"Dataset/FF++/{dataset}/test"

    exp.log_parameters(parameters)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    input_size = 768 * 2 if "DINO" in train_dataset_path else 768
    hidden_size = 512
    final_hidden_size = 256
    num_classes = 2 if one_vs_rest else len(classes)
    learning_rate = lr
    epochs = 300

    # Initialize model, loss function, and optimizer
    model = ClassificationMLP(
        input_size, final_hidden_size, hidden_size, num_classes
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Load datasets and dataloaders
    dataset_train = FlattenedMLPDataset(train_dataset_path, classes, one_vs_rest)
    training_dataloader = DataLoader(
        dataset_train,
        batch_size=batch_train,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=num_workers,
    )

    dataset_val = FastDataset(val_dataset_path, classes, one_vs_rest)
    val_dataloader = DataLoader(
        dataset_val,
        batch_size=val_test_batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=num_workers,
    )

    dataset_test = FastDataset(test_dataset_path, classes, one_vs_rest)
    test_dataloader = DataLoader(
        dataset_test,
        batch_size=val_test_batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=num_workers,
    )

    # Training loop
    for epoch in range(1, epochs + 1):
        train_loss = train(model, training_dataloader, criterion, optimizer, device)
        exp.log_metric(str_classes + " loss", train_loss, step=epoch)
        val_loss, val_accuracy = evaluate_with_voting(
            model, val_dataloader, criterion, device
        )

        exp.log_metric(str_classes + " val_loss", val_loss, step=epoch)
        exp.log_metric(str_classes + " Validation Accuracy", val_accuracy, step=epoch)

        print(
            f"Epoch {epoch}/{epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_accuracy*100:.2f}%"
        )
        scheduler.step()

    # Final testing
    test_loss, test_accuracy = evaluate_with_voting(
        model, test_dataloader, criterion, device
    )
    exp.log_metric(str_classes + " Test Accuracy", test_accuracy)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy*100:.2f}%")


if __name__ == "__main__":
    main()
