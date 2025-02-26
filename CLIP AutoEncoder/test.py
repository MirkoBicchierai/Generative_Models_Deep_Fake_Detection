def test(model, train_loader, test_loader, device):
    # Calculate error on real images (training) to establish baseline statistics
    real_errors = reconstruction_error(model, train_loader, device)
    mean_error = np.mean(real_errors)
    std_error = np.std(real_errors)

    # Calculate error on test images (both real and fake)
    test_errors, labels = reconstruction_error_test(model, test_loader, device)
    labels = np.array(labels)
    true_labels = (labels >= 1).astype(int)  # Fake = 1, Real = 0

    # Create a grid of thresholds to test
    thresholds = np.arange(mean_error, mean_error + 5 * std_error, std_error)
    best_accuracy = 0
    best_threshold = None
    best_predictions = None

    # Find the optimal threshold
    for threshold in thresholds:
        predictions = test_errors > threshold  # True = fake, False = real
        predicted_labels = predictions.astype(int)
        accuracy = accuracy_score(true_labels, predicted_labels)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
            best_predictions = predicted_labels

    # Report final results
    print(f"\nBest threshold: {best_threshold:.4f}")
    print(f"Accuracy: {best_accuracy:.2f}")
    print(f"Percentage of images classified as fake: {np.mean(best_predictions) * 100:.2f}%")

    return best_accuracy

def reconstruction_error(model, dataloader, device):
    model.eval()
    errors = []
    with torch.no_grad():
        for inputs,_ in dataloader:
            inputs = inputs.to(device)
            output, mu, logvar, classification = model(inputs)
            loss = torch.mean((inputs - output) ** 2, dim=1)  # MSE per ogni immagine
            errors.extend(loss.cpu().numpy())
    return np.array(errors)

def reconstruction_error_test(model, dataloader, device):
    model.eval()
    labels_v = []
    losses = []
    with torch.no_grad():
        for sequences, labels in dataloader:
            sequences, labels = sequences.to(device), labels.to(device)

            batch_max_losses = []
            batch_max_frame_indices = []

            # Iterate over each sequence in the batch
            for i, sequence in enumerate(sequences):
                # sequence shape: [10, 768] for 10 frames per video
                frame_losses = []

                # Process each frame individually
                for frame_idx, frame in enumerate(sequence):
                    frame_input = frame.unsqueeze(0)
                    output, mu, logvar, classification = model(frame_input)
                    frame_loss = torch.mean((frame_input - output) ** 2, dim=1)  # MSE per ogni immagine
                    frame_losses.append(frame_loss.item())

                # Find the frame with maximum loss in this sequence
                max_loss = max(frame_losses)
                mean_loss = np.mean(frame_losses)
                max_frame_idx = frame_losses.index(max_loss)

                batch_max_losses.append(max_loss)
                batch_max_frame_indices.append(max_frame_idx)

                losses.append(max_loss)
                labels_v.append(labels[i].item())
                #print(f"Sequence {i}, Label: {labels[i].item()}, Max Loss: {max_loss:.6f}, Frame Index: {max_frame_idx}, Mean loss: {mean_loss}")

    return losses, labels_v
