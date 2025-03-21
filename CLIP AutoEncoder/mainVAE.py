import comet_ml
import numpy as np
import torch
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader
from Model import VariationalAutoencoder
from DataLoader import FastDataset, FlattenedMLPDataset
from tqdm import tqdm
import torch.nn.functional as F


def vae_loss(x_recon, x, mu, logvar, beta=0.75):
    recon_loss = F.l1_loss(x_recon, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_div


def test_vae_classifier(model, test_loader, device):
    model.eval()

    with torch.no_grad():
        errors = []
        true_labels = []

        for data, label in test_loader:
            data, label = data.to(device).squeeze(), label.to(device)
            recon_seq, _, _ = model(data)
            error_recon_sequence = F.mse_loss(recon_seq, data, reduction='sum')
            errors.append(error_recon_sequence.item())
            true_labels.append(label.item())

        distances = np.array(errors)
        true_labels = np.array(true_labels)

        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(true_labels, distances)

        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]

        # Make predictions using optimal threshold
        predictions = (distances >= optimal_threshold).astype(int)

        accuracy = np.mean(predictions == true_labels)

    return accuracy, optimal_threshold


def train(epochs,training_dataloader, val_dataloader, optimizer, model, device, exp):
    for epoch in tqdm(range(epochs)):
        tot_loss = 0
        for sequences, labels in training_dataloader:
            optimizer.zero_grad()
            sequences, labels = sequences.to(device), labels.to(device)
            output, mu, logvar = model(sequences)
            loss = vae_loss(output, sequences, mu, logvar)
            tot_loss += loss.item()
            loss.backward()
            optimizer.step()

        epoch_loss = tot_loss / len(training_dataloader)
        if epoch % 10 == 0:
            print(f"Epoch {epoch} Loss: {epoch_loss:.6f}")
            accuracy, threshold = test_vae_classifier(model, val_dataloader, device)
            exp.log_metric('Validation Accuracy VAE', accuracy, step=epoch)
            print(f"Epoch {epoch} Validation Accuracy: {accuracy:.6f}, Threshold: {threshold:.6f}")
        exp.log_metric('loss VAE', epoch_loss, step=epoch)


def main():
    comet_ml.login(api_key="S8bPmX5TXBAi6879L55Qp3eWW")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset_path = "../Dataset/FF++/CLIP/train"
    val_dataset_path = "../Dataset/FF++/CLIP/val"
    test_dataset_path = "../Dataset/FF++/CLIP/test"

    lr = 1e-3

    #DataLoader parameters
    batch_train = 64
    num_workers = 12

    epochs = 100

    one_vs_rest = True
    classes_train = ["ORIGINAL"]
    classes_test = ["ORIGINAL", "DF"]  # "F2F", "DF", "FSH", "FS", "NT"

    exp = comet_ml.Experiment(project_name="Generative Models Project Work", auto_metric_logging=False, auto_param_logging=False)
    parameters = {'batch_size': batch_train, 'learning_rate': lr}
    exp.log_parameters(parameters)

    dataset_train = FlattenedMLPDataset(train_dataset_path, classes_train, one_vs_rest)
    training_dataloader = DataLoader(dataset_train, batch_size=batch_train, shuffle=True, drop_last=False,pin_memory=True, num_workers=num_workers)

    dataset_val = FastDataset(val_dataset_path, classes_test, one_vs_rest)
    val_dataloader = DataLoader(dataset_val, batch_size=1, shuffle=True, drop_last=False,pin_memory=True, num_workers=num_workers)

    model = VariationalAutoencoder(input_dim=768, latent_dim=8).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr)

    train(epochs, training_dataloader, val_dataloader, optimizer, model, device, exp)

    torch.save(model.state_dict(), "Models/VAE-075.pth")

if __name__ == '__main__':
    main()