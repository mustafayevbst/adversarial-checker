import os
import torch
import torchattacks

from dataset import get_mnist_loaders
from model import SimpleCNN

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def evaluate(model,loader,device,attack=None):
    """Считает точность, в случае с заданной атакой на видоизмененных данных при отсутствии на чистых"""
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if attack is not None:
            images = attack(images, labels)

        with torch.no_grad():
            outputs = model(images)
            preds=outputs.argmax(dim=1)
            correct += (preds==labels).sum().item()
            total+=labels.size(0)

    return correct/total

def test_model(name, model_path, test_loader, device):
    "загрузка модели с проверкой трех сценариев: чисые, fgsm, pgd"
    print(f"========{name}========")
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    #чистые данные
    clean_acc = evaluate(model,test_loader, device)
    print(f"Clean accuracy: {clean_acc:.4f}")

    #FGSM
    fgsm=torchattacks.FGSM(model, eps=0.3)
    fgsm_acc = evaluate(model, test_loader, device, attack=fgsm)
    print(f"FGSM accuracy: {fgsm_acc:.4f} | ASR: {1-fgsm_acc:.4f}")

    #PGD
    pgd=torchattacks.PGD(model, eps=0.3, alpha=0.03, steps=40)
    pgd_acc = evaluate(model, test_loader, device, attack=pgd)
    print(f"PGD accuracy: {pgd_acc:.4f} | ASR: {1-pgd_acc:.4f}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _, test_loader = get_mnist_loaders(batch_size=64)

    #Чистая модель
    test_model("Vanilla model", os.path.join(BASE_DIR, "model.pth"), test_loader, device)

    #Adversarial модель
    test_model("Adversarial model", os.path.join(BASE_DIR, "model_adv.pth"), test_loader, device)

if __name__ == "__main__":
    main()