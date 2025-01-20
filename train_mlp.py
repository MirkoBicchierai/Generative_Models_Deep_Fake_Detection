import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Model import ClassificationMLP
from DataLoader import FlattenedMLPDataset, FastDataset
from tqdm import tqdm
from torch import optim
import numpy as np

one_vs_rest = False  # if true consider all fake classes as same class

train_dataset_path = "Dataset/FF++/DINO/train"
val_dataset_path = "Dataset/FF++/DINO/val"
test_dataset_path = "Dataset/FF++/DINO/test"


classes = [
    "ORIGINAL",
    # "F2F",
    # "DF",
    # "FSH",
    # "FS",
    "NT",
]  # ["ORIGINAL", "F2F", "DF", "FSH", "FS", "NT"]

batch_train = 64
val_test_batch_size = 128
num_workers = 12


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


# Main function
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    input_size = 768 * 2 if "DINO" in train_dataset_path else 768
    hidden_size = 512
    final_hidden_size = 256
    num_classes = 1 if one_vs_rest else len(classes)
    learning_rate = 0.001
    epochs = 150

    # Initialize model, loss function, and optimizer
    model = ClassificationMLP(
        input_size, final_hidden_size, hidden_size, num_classes
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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
        val_loss, val_accuracy = evaluate_with_voting(
            model, val_dataloader, criterion, device
        )

        print(
            f"Epoch {epoch}/{epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_accuracy*100:.2f}%"
        )

    # Final testing
    test_loss, test_accuracy = evaluate_with_voting(
        model, test_dataloader, criterion, device
    )
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy*100:.2f}%")


if __name__ == "__main__":
    main()
