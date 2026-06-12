import torch

def evaluate_model(model, test_loader, model_path):

    # CIFAR-10 class names
    classes = [
        'airplane',
        'automobile',
        'bird',
        'cat',
        'deer',
        'dog',
        'frog',
        'horse',
        'ship',
        'truck'
    ]

    # Device setup
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load saved weights
    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )

    model.to(device)

    # Evaluation mode
    model.eval()

    total = 0
    correct = 0

    class_correct = [0] * 10
    class_total = [0] * 10

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            _, predicted = torch.max(outputs, dim=1)

            # Overall accuracy
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Per-class accuracy
            for label, pred in zip(labels, predicted):

                class_total[label.item()] += 1

                if label == pred:
                    class_correct[label.item()] += 1

    # Overall accuracy
    overall_accuracy = 100 * correct / total

    print(f"\nOverall Test Accuracy: {overall_accuracy:.2f}%\n")

    print("Per-Class Accuracy:")
    print("-" * 30)

    for i in range(10):

        if class_total[i] > 0:
            accuracy = (
                100 * class_correct[i] / class_total[i]
            )
        else:
            accuracy = 0

        print(
            f"{classes[i]:<12}: "
            f"{accuracy:.2f}%"
        )
    
    return overall_accuracy