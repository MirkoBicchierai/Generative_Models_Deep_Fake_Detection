import comet_ml
import torch
from torch import nn
from torch.utils.data import DataLoader
from DataLoader import FastDataset
from tqdm import tqdm
from torchmetrics import Accuracy
from Model import ClassificationLSTM, ClassificationGRU
import numpy as np
from oscr import compute_oscr

all_classes = [
    "ORIGINAL",
    "F2F",
    "DF",
    "FSH",
    "FS",
    "NT",
]

# nicco api key
# comet_ml.login(api_key="WQRfjlovs7RSjYUmjlMvNt3PY")

# mirko api
comet_ml.login(api_key="S8bPmX5TXBAi6879L55Qp3eWW")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = "CLIP"
    epochs = 300

    if dataset == "CLIP":
        train_dataset_path = "Dataset/FF++/CLIP/train"
        val_dataset_path = "Dataset/FF++/CLIP/val"
        test_dataset_path = "Dataset/FF++/CLIP/test"
        input_size = 768

        # SGD and scheduler Parameters
        lr = 1e-3
        momentum = 0.9
        weight_decay = 0.0001

        # RNN parameters
        layers = 3
        hidden_size = 512

    else:
        train_dataset_path = "Dataset/FF++/DINO/train"
        val_dataset_path = "Dataset/FF++/DINO/val"
        test_dataset_path = "Dataset/FF++/DINO/test"
        input_size = 1536

        # SGD and scheduler Parameters
        lr = 1e-2
        momentum = 0.9
        weight_decay = 0.0001

        # RNN parameters
        layers = 3
        hidden_size = 512

    # DataLoader parameters
    batch_train = 64
    val_test_batch_size = 64
    num_workers = 12

    # Training type parameters
    classes = [
        "ORIGINAL",
        "F2F",
        "DF",
        "FSH",
        "FS",
        # "NT",
    ]  # ["ORIGINAL", "F2F", "DF", "FSH", "FS", "NT"]
    oscr_classes = list(set(all_classes) - set(classes))
    lstm = False
    one_vs_rest = True  # if true consider all fake classes as same class
    if one_vs_rest:
        # str_classes = "-".join(classes[1:])
        # str_classes = "ORIGINAL vs " + str_classes
        str_classes = "Missing " + oscr_classes[0]

    else:
        str_classes = "-".join(classes)

    exp = comet_ml.Experiment(
        project_name="Generative Models Project Work",
        auto_metric_logging=False,
        auto_param_logging=False,
    )
    exp.set_name(dataset + " - " + "GRU" + " - " + str_classes)
    parameters = {
        "batch_size": batch_train,
        "learning_rate": lr,
        "hidden_size": hidden_size,
        "layers": layers,
        "type": str_classes,
    }
    exp.log_parameters(parameters)

    oscr_dataset = FastDataset(val_dataset_path, oscr_classes, one_vs_rest)
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

    if lstm:
        model = ClassificationLSTM(
            hidden_size, input_size, dataset_train.num_classes, layers
        ).to(device)
    else:
        model = ClassificationGRU(
            hidden_size, input_size, dataset_train.num_classes, layers
        ).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(), lr, momentum=momentum, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    loss_fn = nn.CrossEntropyLoss()
    accuracy = Accuracy(task="multiclass", num_classes=dataset_train.num_classes).to(
        device
    )

    for epoch in tqdm(range(epochs)):
        tot_loss = 0
        for sequences, labels in training_dataloader:
            optimizer.zero_grad()
            sequences, labels = sequences.to(device), labels.to(device)
            logit = model(sequences)
            loss = loss_fn(logit, labels)
            tot_loss += loss.item()
            loss.backward()
            optimizer.step()

        scheduler.step()
        epoch_loss = tot_loss / len(training_dataloader)
        exp.log_metric(str_classes + " loss", epoch_loss, step=epoch)

        """ VALIDATION"""
        model.eval()
        tot_acc_val = 0
        tot_val_loss = 0
        for sequences, labels in val_dataloader:
            sequences, labels = sequences.to(device), labels.to(device)
            with torch.no_grad():
                logit = model(sequences)
                loss = loss_fn(logit, labels)
                tot_val_loss += loss.item()
                tot_acc_val += accuracy(logit, labels).item()

        tot_val_loss = tot_val_loss / len(val_dataloader)
        exp.log_metric(str_classes + " val_loss", tot_val_loss, step=epoch)
        val_acc = tot_acc_val / len(val_dataloader)
        exp.log_metric(str_classes + " Validation Accuracy", val_acc, step=epoch)

        print(
            "Epoch:",
            epoch + 1,
            "Training loss:",
            epoch_loss,
            "Validation Loss:",
            tot_val_loss,
            "Validation Accuracy:",
            round(val_acc * 100, 2),
        )
        model.train()

    """TEST"""
    model.eval()
    tot_acc_test = 0
    all_labels = []
    all_probs = []
    for sequences, labels in test_dataloader:
        sequences, labels = sequences.to(device), labels.to(device)
        all_labels.append(labels.cpu().numpy().tolist())
        with torch.no_grad():
            logit = model(sequences)
            probs = torch.nn.functional.softmax(logit, dim=1)
        tot_acc_test += accuracy(logit, labels).item()
        all_probs.append(probs.cpu().numpy())

    test_acc = tot_acc_test / len(test_dataloader)
    exp.log_metric(str_classes + " Test Accuracy", test_acc)
    print("Test Accuracy:", round(test_acc * 100, 2))
    all_labels = np.concatenate(all_labels)
    pred_k = np.concatenate(all_probs)

    """ get predictions for unknown classes to compute OSCR """
    all_probs = []
    for sequences, labels in oscr_dataloader:
        sequences, labels = sequences.to(device), labels.to(device)
        with torch.no_grad():
            logit = model(sequences)
            probs = torch.nn.functional.softmax(logit, dim=1)
        tot_acc_test += accuracy(logit, labels).item()
        all_probs.append(probs.cpu().numpy())
    pred_u = np.concatenate(all_probs)

    results = compute_oscr(pred_k, pred_u, all_labels)
    print(f"OSCR Score (AUROC): {results['oscr']:.4f}")
    print(f"CCR @ 5% FPR: {results['ccr@fpr05']:.4f}")
    exp.log_metric("OSCR", results["oscr"])
    exp.log_metric("CCR @ 5% FPR", results["ccr@fpr05"])


if __name__ == "__main__":
    main()
