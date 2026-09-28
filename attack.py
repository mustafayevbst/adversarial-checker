import os
import torch
import torch.nn as nn
import torchattacks
import matplotlib.pyplot as plt

from dataset import get_mnist_loaders
from model import SimpleCNN

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def evaluate(model, loader, device, attack=None):
    """Считает accuracy и ASR. Если attack=None то на чистых данных."""
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if attack is not None:
            images = attack(images, labels)

        with torch.no_grad():
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Данные
    _, test_loader = get_mnist_loaders(batch_size=64)

    # Модель
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(os.path.join(BASE_DIR, "model.pth"), map_location=device))
    print("Model loaded")

    #точность на чистых данных
    clean_acc = evaluate(model, test_loader, device)
    print(f"Clean accuracy: {clean_acc:.4f}")

    # FGSM
    fgsm = torchattacks.FGSM(model, eps=0.3)
    fgsm_acc = evaluate(model, test_loader, device, attack=fgsm)
    print(f"FGSM  accuracy: {fgsm_acc:.4f}  |  ASR: {1 - fgsm_acc:.4f}")

    #  PGD
    pgd = torchattacks.PGD(model, eps=0.3, alpha=0.03, steps=40)
    pgd_acc = evaluate(model, test_loader, device, attack=pgd)
    print(f"PGD   accuracy: {pgd_acc:.4f}  |  ASR: {1 - pgd_acc:.4f}")

    #визуализация в виде оригинал / шум / атакованная картинка
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)

    adv_fgsm = fgsm(images, labels)
    adv_pgd = pgd(images, labels)

    # первая картинка
    img = images[0].cpu().squeeze()
    noise_fgsm = (adv_fgsm[0] - images[0]).cpu().squeeze()
    adv_img_fgsm = adv_fgsm[0].cpu().squeeze()

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title(f"Original: {labels[0].item()}")
    axes[1].imshow(noise_fgsm, cmap="gray")
    axes[1].set_title("FGSM noise")
    axes[2].imshow(adv_img_fgsm, cmap="gray")
    axes[2].set_title("Adversarial")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    out_path = os.path.join(BASE_DIR, "attack_example.png")
    plt.savefig(out_path)
    print(f"Saved visualization to {out_path}")


if __name__ == "__main__":
    main()