import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import comet_ml
import torch
from torch.utils.data import DataLoader
from DataLoader import FastDataset
from tqdm import tqdm
from Model import TransformerClassifier
import torch.nn.functional as F
from oscr import compute_oscr
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

all_classes = [
    "ORIGINAL",
    "F2F",
    "DF",
    "FSH",
    "FS",
    "NT",
]  # ["ORIGINAL", "F2F", "DF", "FSH", "FS", "NT"]


def criterion(logits, aux_logits, targets, alpha=0.3):
    main_loss = F.cross_entropy(logits, targets)
    aux_loss = F.cross_entropy(aux_logits, targets)
    total_loss = (1 - alpha) * main_loss + alpha * aux_loss
    return total_loss


def test(model, data_loader, device):
    model.eval()
    loss = 0.0
    correct = 0
    total = 0
    all_outputs = []
    all_labels = []
    with torch.no_grad():
        for sequences, labels in data_loader:
            all_labels.append(labels.cpu().numpy().tolist())
            sequences, labels = sequences.to(device), labels.to(device)
            logits, axu_logits = model(sequences)
            loss = criterion(logits, axu_logits, labels)
            loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            probs = torch.nn.functional.softmax(logits, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_outputs.append(probs.cpu().numpy())

    return (
        loss / len(data_loader),
        correct / total,
        np.concatenate(all_outputs),
        np.concatenate(all_labels),
    )


def train(
    epochs,
    model,
    optimizer,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    oscr_dataloader,
    device,
    exp,
    str_classes,
):
    for epoch in tqdm(range(epochs)):
        model.train()
        tot_loss = 0
        for sequences, labels in train_dataloader:
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, axu_logits = model(sequences)
            loss = criterion(logits, axu_logits, labels)
            tot_loss += loss.item()
            loss.backward()
            optimizer.step()

        epoch_loss = tot_loss / len(train_dataloader)
        exp.log_metric("loss Transformer", epoch_loss, step=epoch)

        if epoch % 10 == 0:
            val_loss, val_accuracy, _, _ = test(model, val_dataloader, device)
            print(f"Epoch {epoch} Train Loss: {epoch_loss:.6f}")
            print(f"Val Loss: {val_loss:.6f}, Val Accuracy: {val_accuracy:.6f}")
            exp.log_metric(
                str_classes + "Validation accuracy Transformer", val_accuracy
            )
            exp.log_metric(
                str_classes + "Validation loss Transformer", val_loss, step=epoch
            )

    test_loss, test_accuracy, pred_k, labels = test(model, test_dataloader, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy}")
    exp.log_metric(str_classes + "Test accuracy Transformer", test_accuracy)
    _, _, pred_u, _ = test(model, oscr_dataloader, device)

    results = compute_oscr(pred_k, pred_u, labels)
    print(f"OSCR Score (AUROC): {results['oscr']:.4f}")
    print(f"CCR @ 5% FPR: {results['ccr@fpr05']:.4f}")
    exp.log_metric("OSCR", results["oscr"])
    exp.log_metric("CCR @ 5% FPR", results["ccr@fpr05"])


def main():
    # mirko api
    comet_ml.login(api_key="S8bPmX5TXBAi6879L55Qp3eWW")
    # nicco api
    # comet_ml.login(api_key="WQRfjlovs7RSjYUmjlMvNt3PY")

    train_dataset_path = "../Dataset/FF++/CLIP/train"
    val_dataset_path = "../Dataset/FF++/CLIP/val"
    test_dataset_path = "../Dataset/FF++/CLIP/test"
    input_dim = 768

    lr = 1e-4

    # DataLoader parameters
    batch_train = 64
    val_test_batch_size = 64
    num_workers = 12

    epochs = 150

    one_vs_rest = False
    classes = [
        "ORIGINAL",
        # "F2F",
        "DF",
        "FSH",
        "FS",
        "NT",
    ]  # ["ORIGINAL", "F2F", "DF", "FSH", "FS", "NT"]
    missing_classes = list(set(all_classes) - set(classes))
    if len(missing_classes) == 1:
        str_classes = "Missing " + missing_classes[0]
    elif one_vs_rest:
        str_classes = "-".join(classes[1:])
        str_classes = "ORIGINAL vs " + str_classes
    else:
        str_classes = "-".join(classes)

    exp = comet_ml.Experiment(
        project_name="Generative Models Project Work",
        auto_metric_logging=False,
        auto_param_logging=False,
    )

    exp.set_name("TRANSFORMER " + " - " + "CLIP" + " - " + str_classes)
    parameters = {"batch_size": batch_train, "learning_rate": lr}
    exp.log_parameters(parameters)

    oscr_dataset = FastDataset(val_dataset_path, missing_classes, one_vs_rest)
    oscr_dataloader = DataLoader(
        oscr_dataset,
        batch_size=val_test_batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=num_workers,
    )

    dataset_train = FastDataset(train_dataset_path, classes, one_vs_rest)
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
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=num_workers,
    )

    dataset_test = FastDataset(test_dataset_path, classes, one_vs_rest)
    test_dataloader = DataLoader(
        dataset_test,
        batch_size=val_test_batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=num_workers,
    )

    model = TransformerClassifier(
        input_dim=input_dim,
        hidden_dim=256,
        num_classes=dataset_train.num_classes,
        num_heads=16,
        num_layers=8,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr)

    train(
        epochs,
        model,
        optimizer,
        training_dataloader,
        val_dataloader,
        test_dataloader,
        oscr_dataloader,
        device,
        exp,
        str_classes,
    )


if __name__ == "__main__":
    main()

