import os
import time
import argparse
import torch
import torchattacks

from dataset import get_mnist_loaders
from model import SimpleCNN

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def evaluate(model, loader, device, attack= None):
    "считает точность, если атака то на атак данных"

    model.eval()
    correct, total= 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if attack is not None:
            images = attack(images, labels)

        with torch.no_grad():
            outputs = model(images)
            preds=outputs.argmax(dim=1)
            correct+=(preds==labels).sum().item()
            total+=labels.size(0)

    return correct/total

def load_model(model_path, device):
    "выгрузка в simplecnn весов"
    model=SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

def build_attack(name, model, eps, steps):
    "вернет какую атаку выполнит если без атаки то None"
    if name == "none":
        return None
    if name == "fgsm":
        return torchattacks.FGSM(model, eps=eps)
    if name == "pgd":
        return torchattacks.PGD(model, eps=eps, alpha= eps/10, steps=steps)
    raise ValueError(f"Unknown attack: {name}")

def main():
    parser = argparse.ArgumentParser( description="Adversarial Robustness Checker / Оценка устойчивост ML-модели к атаками")
    parser.add_argument("--model", type=str, default="model.pth",
                         help="Путь к файлу модели (.pth)")
    parser.add_argument("--attack", type=str, default="fgsm",
                         choices=["fgsm", "pgd", "none"],
                         help = "Какую атаку применить (fgsm/pgd/none)")
    parser.add_argument("--eps", type=float, default=0.3,
                        help="Максималльное возмущение (epsilon)")
    parser.add_argument("--steps", type=int, default=40,
                        help="Число шагов для PGD")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Размер батча для теста")
    args=parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path=os.path.join(BASE_DIR, args.model)
    if not os.path.exists(model_path):
        print(f"[ERROR] Модель не найдена: {model_path}")
        return
    model = load_model(model_path, device)

    _, test_loader = get_mnist_loaders(batch_size=args.batch_size)

    print(f"===Model: {args.model}===")
    print(f"Device: {device}")

    t0=time.time()
    clean_acc=evaluate(model, test_loader, device)
    print(f"Clean accuracy: {clean_acc:.4f}")

    if args.attack == "none":
        print("Attack: none (Только чистая точность)")
    else:
        attack=build_attack(args.attack, model, args.eps, args.steps)
        atk_acc=evaluate(model, test_loader, device, attack=attack)
        print(f"Attack: {args.attack.upper()} (eps={args.eps}, steps={args.steps})")
        print(f"Attacked accuracy: {atk_acc:.4f}")
        print(f"ASR: {1- atk_acc:.4f}")

    elapsed = time.time() - t0
    print(f"Time: {elapsed:.1f}s")

if __name__ == "__main__":
    main()