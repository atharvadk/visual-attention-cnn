import torch
import torch.nn as nn
import torch.optim as optim


def train_model(model, train_loader, test_loader, epochs=30):

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = model.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs
    )

    # Training loop
    for epoch in range(epochs):

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Loss
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()

            # Statistics
            running_loss += loss.item()

            _, predicted = torch.max(outputs, dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Epoch metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total

        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Loss: {epoch_loss:.4f} | "
            f"Train Accuracy: {epoch_acc:.2f}%"
        )

        # Step scheduler
        scheduler.step()

    return model