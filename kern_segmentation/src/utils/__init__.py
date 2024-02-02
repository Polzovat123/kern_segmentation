import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from .Cutter import TresholdCutter

def calculate_accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1)
    correct_pixels = torch.eq(predictions, targets).sum().item()
    total_pixels = targets.numel()
    accuracy = correct_pixels / total_pixels
    return accuracy

def calculate_iou(predictions, targets):
    intersection = torch.logical_and(targets, predictions).sum().item()
    union = torch.logical_or(targets, predictions).sum().item()
    iou = intersection / union
    return iou

def log_metrics(phase, epoch, loss, accuracy, iou):
    print(f'{phase}_loss', loss, epoch)
    print(f'{phase}_accuracy', accuracy, epoch)
    print(f'{phase}_iou', iou, epoch)

def train(model, train_loader, criterion, optimizer, epoch, log_metrics, save_folder):
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    total_iou = 0.0

    for ep in tqdm(range(1, epoch)):
      for (inputs, targets) in tqdm(train_loader):
          inputs = inputs.unsqueeze(0).permute(0, 3, 1, 2)
          targets = targets.unsqueeze(0).unsqueeze(-1).permute(0, 3, 1, 2)

          optimizer.zero_grad()
          outputs = model(inputs)

          loss = criterion(outputs[0, 0, :, :], targets[0, 0, :, :])
          loss.backward()
          optimizer.step()

          total_loss += loss.item()
          accuracy = calculate_accuracy(outputs, targets)
          total_accuracy += accuracy
          iou = calculate_iou(outputs, targets)
          total_iou += iou

          # progress_bar.set_postfix({'Loss': loss.item(), 'Accuracy': accuracy, 'IOU': iou})

          average_loss = total_loss / len(train_loader)
          average_accuracy = total_accuracy / len(train_loader)
          average_iou = total_iou / len(train_loader)

          # log_metrics('train', epoch, average_loss, average_accuracy, average_iou)

          # Save the model if it has the best performance so far
          if not os.path.exists(save_folder):
              os.makedirs(save_folder)
          save_path = f"{save_folder}/model_epoch_{epoch}.pt"
          torch.save(model.state_dict(), save_path)
          # break

    return save_path

def validate(model, val_loader, criterion, epoch):
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    total_iou = 0.0

    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            accuracy = calculate_accuracy(outputs, targets)
            total_accuracy += accuracy
            iou = calculate_iou(outputs, targets)
            total_iou += iou

    average_loss = total_loss / len(val_loader)
    average_accuracy = total_accuracy / len(val_loader)
    average_iou = total_iou / len(val_loader)

    log_metrics('val', epoch, average_loss, average_accuracy, average_iou)

#%%
def visualize_random_sample(model, train_loader):
    random_index = np.random.randint(len(train_loader.dataset))
    sample = train_loader.dataset[random_index]

    # print(sample)
    input_image, target_mask = sample[0], sample[1]

    # Add batch dimension and adjust dimensions
    input_image = input_image.unsqueeze(0).permute(0, 3, 1, 2)
    target_mask = target_mask.unsqueeze(0).unsqueeze(-1).permute(0, 3, 1, 2)

    # Forward pass to get model predictions
    model.eval()
    with torch.no_grad():
        predicted_mask = model(input_image)

    # Plot the original image, ground truth mask, and predicted mask
    print(random_index)
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(input_image[0, 0, :, :])
    plt.title('Original Image')
    plt.axis('off')

    print(target_mask.shape)
    plt.subplot(1, 3, 2)
    plt.imshow(target_mask[0, 0, :, :])
    plt.title('Ground Truth Mask')
    plt.axis('off')

    print(predicted_mask.shape)
    plt.subplot(1, 3, 3)
    plt.imshow(predicted_mask.squeeze())
    plt.title('Predicted Mask')
    plt.axis('off')

    plt.show()