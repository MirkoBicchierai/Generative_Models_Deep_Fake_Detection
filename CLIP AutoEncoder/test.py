import numpy as np
import torch
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader
from Model import AutoEncoder, VariationalAutoencoder
from DataLoader import FastDataset
import torch.nn.functional as F

def test_ae_classifier(model, test_loader, device, vae):
    model.eval()

    with torch.no_grad():

        errors = []
        true_labels = []

        for data, label in test_loader:
            data, label = data.to(device).squeeze(), label.to(device)
            if vae:
                recon_seq, _, _ = model(data)
            else:
                recon_seq = model(data)
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

def main():

    vae = True
    model_path = "../CLIP AutoEncoder/Models/VAE-1.pth"

    one_vs_rest = True
    classes_test = ["ORIGINAL", "DF"]  # "F2F", "DF", "FSH", "FS", "NT"

    test_dataset_path = "../Dataset/FF++/CLIP/test"

    dataset_test = FastDataset(test_dataset_path, classes_test, one_vs_rest)
    test_dataloader = DataLoader(dataset_test, batch_size=1, shuffle=True, drop_last=False, pin_memory=True,
                                 num_workers=12)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if vae:
        model = VariationalAutoencoder(input_dim=768, latent_dim=8).to(device)
    else:
        model = AutoEncoder(input_dim=768).to(device)

    model.load_state_dict(torch.load(model_path))
    model.eval()

    accuracy, threshold = test_ae_classifier(model, test_dataloader, device, vae)
    print(f"Test Accuracy: {accuracy:.6f}, Threshold: {threshold:.6f}")

if __name__ == '__main__':
    main()

