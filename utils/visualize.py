import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

def get_attention_map(model, image_tensor):
    """
    Extract spatial attention map from the last CNNBlock's CBAM.
    """

    device = next(model.parameters()).device
    attention_map = None

    def hook_fn(module, inputs, output):
        nonlocal attention_map

        # output shape: (1, C, H, W)
        # Recover spatial attention by comparing output to input
        x = inputs[0]

        with torch.no_grad():
            attention = output / (x + 1e-8)

            # average over channels
            attention = attention.mean(dim=1)

            attention_map = attention.squeeze().cpu().numpy()

    # Register hook on LAST CNNBlock's spatial attention
    hook = model.layer3.cbam.spatial_attention.register_forward_hook(
        hook_fn
    )

    model.eval()

    with torch.no_grad():
        _ = model(image_tensor.unsqueeze(0).to(device))

    hook.remove()

    return attention_map

def visualize(model, test_loader, num_images=5):

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

    device = next(model.parameters()).device

    model.eval()

    shown = 0

    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])

    with torch.no_grad():

        for images, labels in test_loader:

            for i in range(images.size(0)):

                if shown >= num_images:
                    return

                image = images[i]
                label = labels[i].item()

                # prediction
                output = model(
                    image.unsqueeze(0).to(device)
                )

                pred = output.argmax(dim=1).item()

                # attention map
                attention_map = get_attention_map(
                    model,
                    image
                )

                # convert image back for display
                img = image.permute(1, 2, 0).cpu().numpy()

                img = img * std + mean
                img = np.clip(img, 0, 1)

                plt.figure(figsize=(10, 4))

                # Original image
                plt.subplot(1, 2, 1)
                plt.imshow(img)
                plt.axis("off")
                plt.title("Original")

                # Overlay attention map
                attention_resized = cv2.resize(
                    attention_map, 
                    (32, 32), 
                    interpolation=cv2.INTER_LINEAR
                )

                # Normalize to 0-1
                attention_resized = (attention_resized - attention_resized.min()) / \
                                    (attention_resized.max() - attention_resized.min() + 1e-8)

                # Overlay attention map
                plt.subplot(1, 2, 2)
                plt.imshow(img)
                plt.imshow(
                    attention_resized,
                    cmap="jet",
                    alpha=0.4
                )

                plt.axis("off")

                plt.title(
                    f"Pred: {classes[pred]}\n"
                    f"True: {classes[label]}"
                )

                plt.tight_layout()
                plt.show()

                shown += 1