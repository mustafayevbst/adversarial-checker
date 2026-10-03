import os 
import pytest
import torch
import torchattacks

import sys

#нужно для импортирования модулей проекта
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import SimpleCNN
from dataset import get_mnist_loaders

@pytest.fixture
def model():
    return SimpleCNN()

@pytest.fixture
def sample_batch():
    #реальный батч MNIST
    train_loader, _ = get_mnist_loaders(batch_size=4)
    images, labels = next(iter(train_loader))
    return images, labels

def test_model_output_shape(model, sample_batch):
    images, _=sample_batch
    output = model(images)
    assert output.shape ==(4,10), f"Ожидалось (4,10), было получено {output.shape}"

def test_model_output_is_finite(model, sample_batch):
    images, _=sample_batch
    output=model(images)
    assert torch.isfinite(output).all(), "В логитах есть NaN или inf"

def test_fgsm_changes_images(model, sample_batch):
    images, labels=sample_batch
    attack= torchattacks.FGSM(model, eps=0.3)
    adv_images=attack(images,labels)
    assert not torch.equal(adv_images, images), "FGSM не изменил картинки"

def test_fgsm_perturbation_within_eps(model, sample_batch):
    images, labels = sample_batch
    eps=0.3
    attack=torchattacks.FGSM(model, eps=eps)
    adv_images=attack(images,labels)
    max_diff=(adv_images - images).abs().max().item()
    assert max_diff<=eps+1e-6, f"Возмущение {max_diff} больше eps={eps}"

def test_dataloader_shapes():
    train_loader, _=get_mnist_loaders(batch_size=32)
    images, labels = next(iter(train_loader))
    assert images.shape== (32, 1, 28, 28)
    assert labels.shape == (32,)
    assert images.min() >=0.0 and images.max()<=1.0