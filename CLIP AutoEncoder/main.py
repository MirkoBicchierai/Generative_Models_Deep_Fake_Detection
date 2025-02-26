import comet_ml
import torch
from torch.utils.data import DataLoader
from Model import AutoEncoder, VariationalAutoencoder
from DataLoader import FastDataset, FlattenedMLPDataset
from tqdm import tqdm
import torch.nn.functional as F


def vae_loss(x_recon, x, mu, logvar, classification, target_class, beta=0.1, alpha=0.94):
    recon_loss = F.l1_loss(x_recon, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    class_loss = F.cross_entropy(classification, target_class)
    return 0.05 * recon_loss + beta * kl_div + alpha * class_loss


def test_vae_classifier(model, test_loader, device):
    model.eval()
    correct_sequences = 0
    total_sequences = 0
    with torch.no_grad():
        for data, label in test_loader:
            label = label.to(device)
            data, labels = data.to(device).squeeze(), label.to(device).repeat(10)

            recon_batch, mu, logvar, classification = model(data)

            loss = F.cross_entropy(classification, labels)

            predicted_frames = classification.argmax(dim=1)  # Shape: [10]

            # Votazione a maggioranza
            voted_prediction = torch.mode(predicted_frames).values  # Classe più frequente

            # Confronto con la label della sequenza
            if voted_prediction == label:
                correct_sequences += 1
            total_sequences += 1

        # Calcoliamo l'accuratezza sulle sequenze testate
        sequence_accuracy = correct_sequences / total_sequences
        print(f"Sequence Accuracy: {sequence_accuracy:.4f}")

    return


def main():
    comet_ml.login(api_key="S8bPmX5TXBAi6879L55Qp3eWW")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset_path = "../Dataset/FF++/CLIP/train"
    val_dataset_path = "../Dataset/FF++/CLIP/val"
    test_dataset_path = "../Dataset/FF++/CLIP/test"

    lr = 1e-3

    #DataLoader parameters
    batch_train = 64
    val_test_batch_size = 16
    num_workers = 12

    epochs = 500

    one_vs_rest = False
    classes = ["ORIGINAL", "F2F"] # , "DF", "FSH", "FS", "NT"

    exp = comet_ml.Experiment(project_name="Generative Models Project Work", auto_metric_logging=False, auto_param_logging=False)
    parameters = {'batch_size': batch_train, 'learning_rate': lr}
    exp.log_parameters(parameters)

    dataset_train = FlattenedMLPDataset(train_dataset_path, classes, one_vs_rest)
    num_classes = dataset_train.num_classes
    training_dataloader = DataLoader(dataset_train, batch_size=batch_train, shuffle=True, drop_last=False,pin_memory=True, num_workers=num_workers)

    dataset_val = FastDataset(val_dataset_path, classes, one_vs_rest)
    val_dataloader = DataLoader(dataset_val, batch_size=1, shuffle=True, drop_last=False,pin_memory=True, num_workers=num_workers)

    dataset_test = FastDataset(test_dataset_path, classes, one_vs_rest)
    test_dataloader = DataLoader(dataset_test, batch_size=1, shuffle=True, drop_last=False,pin_memory=True, num_workers=num_workers)

    model = VariationalAutoencoder(input_dim=768, num_classes= num_classes).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr)

    for epoch in tqdm(range(epochs)):
        tot_loss = 0
        for sequences, labels in training_dataloader:
            optimizer.zero_grad()
            sequences, labels = sequences.to(device), labels.to(device)
            output, mu, logvar, classification = model(sequences)
            loss = vae_loss(output,sequences, mu, logvar, classification, labels)
            tot_loss += loss.item()
            loss.backward()
            optimizer.step()

        epoch_loss = tot_loss / len(training_dataloader)
        if epoch%10 == 0:
            print(f"Epoch {epoch} Loss: {epoch_loss:.6f}")
            test_vae_classifier(model, val_dataloader, device)
        exp.log_metric('loss AE', epoch_loss, step=epoch)

    test_vae_classifier(model, test_dataloader, device)

if __name__ == '__main__':
    main()