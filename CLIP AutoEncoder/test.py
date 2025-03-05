import numpy as np
import torch
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader
from Model import AutoEncoder, VariationalAutoencoder
from DataLoader import FastDataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

def plot_mean_bar(errors_original, errors_fake, fake_class, name):

    # Bar plot
    plt.figure(figsize=(7, 5))

    colors = ["#4CAF50"] + ["#E57373" for _ in fake_class]

    plt.bar(["Original"] + fake_class, [errors_original]+errors_fake, color=colors)
    # Labels and title
    plt.ylabel("Mean of MSE errors")
    plt.title("Mean of MSE errors per Class - " + name)
    plt.savefig("../CLIP AutoEncoder/Plot/bar_plot_"+name+".pdf")

def plot_log_mean_bar(errors_original, errors_fake, fake_class, name):
    # Bar plot
    plt.figure(figsize=(7, 5))

    colors = ["#4CAF50"] + ["#E57373" for _ in fake_class]

    plt.bar(["Original"] + fake_class, [errors_original] + errors_fake, color=colors)
    plt.yscale('log')
    # Labels and title
    plt.ylabel("Mean of MSE errors")
    plt.title("Mean of MSE errors per Class - " + name)
    plt.savefig("../CLIP AutoEncoder/Plot/bar_plot_logscale_" + name + ".pdf")


def plot_histo(errors_original, errors_fake, fake_class, name, bins=20):

    plt.figure(figsize=(10, 6))

    colors = sns.color_palette("husl", len(fake_class))
    for i, (errors, label) in enumerate(zip(errors_fake, fake_class)):
        sns.histplot(errors, bins=bins, color=colors[i], label=label, alpha=0.6, kde=True)

    sns.histplot(errors_original, bins=bins, color='blue', label='Original', alpha=0.4, kde=True)

    plt.xlabel("Reconstruction Error")
    plt.ylabel("Count")
    plt.title("Histogram of Reconstruction Errors")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("../CLIP AutoEncoder/Plot/hist_" + name + ".pdf")


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

        errors_fake = distances[true_labels == 1]
        error_original = distances[true_labels == 0]
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(true_labels, distances)

        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]

        # Make predictions using optimal threshold
        predictions = (distances >= optimal_threshold).astype(int)

        accuracy = np.mean(predictions == true_labels)

    return accuracy, optimal_threshold, errors_fake, error_original


def main():
    vae = True
    name_plot = "VAE beta=0,75"
    model_path = "../CLIP AutoEncoder/Models/VAE-075.pth"
    test_dataset_path = "../Dataset/FF++/CLIP/test"

    one_vs_rest = True
    fake_class = ["F2F", "DF", "FSH", "FS", "NT"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if vae:
        model = VariationalAutoencoder(input_dim=768, latent_dim=8).to(device)
    else:
        model = AutoEncoder(input_dim=768).to(device)

    model.load_state_dict(torch.load(model_path))
    model.eval()

    errors_original_mean = []
    errors_fake_mean = []

    errors_original = []
    errors_fake = []
    for fc in fake_class:

        classes_test = ["ORIGINAL", fc]

        dataset_test = FastDataset(test_dataset_path, classes_test, one_vs_rest)
        test_dataloader = DataLoader(dataset_test, batch_size=1, shuffle=True, drop_last=False, pin_memory=True,
                                     num_workers=12)

        accuracy, threshold, ef, eo = test_ae_classifier(model, test_dataloader, device, vae)
        errors_original_mean = np.mean(eo)
        errors_fake_mean.append(np.mean(ef))
        errors_fake.append(ef)
        errors_original = eo
        print(f"Test Accuracy {fc}: {accuracy:.6f}, Threshold: {threshold:.6f}")


    plot_mean_bar(errors_original_mean, errors_fake_mean, fake_class, name_plot)
    plot_log_mean_bar(errors_original_mean, errors_fake_mean, fake_class, name_plot)
    plot_histo(errors_original, errors_fake, fake_class, name_plot)

if __name__ == '__main__':
    main()
