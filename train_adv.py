import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchattacks

from dataset import get_mnist_loaders
from model import SimpleCNN

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def train_one_epoch_adv(model, loader, optimizer, criterion, device, attack):
    """Одна эпоха adversarial training."""
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        adv_images = attack(images, labels)

        # половина чистых, половина атакованных
        mixed_images = torch.cat([images, adv_images], dim=0)
        mixed_labels = torch.cat([labels, labels], dim=0)

        optimizer.zero_grad()
        outputs = model(mixed_images)
        loss = criterion(outputs, mixed_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * mixed_images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == mixed_labels).sum().item()
        total += mixed_labels.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_mnist_loaders(batch_size=64)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Атака
    attack = torchattacks.FGSM(model, eps=0.3)

    epochs = 5
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch_adv(
            model, train_loader, optimizer, criterion, device, attack
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
              f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}")

    out_path = os.path.join(BASE_DIR, "model_adv.pth")
    torch.save(model.state_dict(), out_path)
    print(f"Saved adversarial model to {out_path}")


if __name__ == "__main__":
    main()