import random

from IPython import display
import numpy as np
import matplotlib.pyplot as plt
import torch
import tqdm
from sklearn.metrics import f1_score, roc_curve, confusion_matrix


def get_accuracy(model, dataloader, device):
    correct = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            prediction = model(x).argmax(dim=-1, keepdim=True)
            correct += prediction.eq(y.view_as(prediction)).sum().item()
    return correct / len(dataloader.dataset)


def f1_score_(model, dataloader, device):
    correct = 0
    predicted_y = np.array([])
    true_y = np.array([])
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            val_pred = model(x)
            predicted_y = np.concatenate((predicted_y, np.argmax(val_pred.cpu().detach(), axis=1)), axis=None)
            true_y = np.concatenate((true_y, y.cpu().detach()), axis=None)
    return f1_score(true_y, predicted_y, average='macro')
  


class Monitor:
    def __init__(self):
        if hasattr(tqdm.tqdm, '_instances'):
            [*map(tqdm.tqdm._decr_instances, list(tqdm.tqdm._instances))]

        self.learning_curve = []
        self.train_accuracy_curve = []
        self.val_accuracy_curve = []
        self.val_f1_curve = []
        
        self.best_val_accuracy = 0
        self.best_val_epoch = 0
        self.best_val_f1 = 0

    def add_loss_value(self, value):
        self.learning_curve.append(value)

    def add_train_accuracy_value(self, value):
        self.train_accuracy_curve.append(value)

    def add_val_accuracy_value(self, value):
        self.val_accuracy_curve.append(value)

        if value > self.best_val_accuracy:
            self.best_val_accuracy = value
            self.best_val_epoch = len(self.val_accuracy_curve)
            
    ######################################################## 
    
    def add_val_f1_value(self, value):
        self.val_f1_curve.append(value)
        
        if value > self.best_val_f1:
            self.best_val_f1 = value
            self.best_val_f1_epoch = len(self.val_f1_curve)
            
    #########################################################        
    
    def show(self):
        display.clear_output(wait=True)
        f, axes = plt.subplots(1, 2, figsize=(15, 3))
        axes[0].set_title('Loss')
        axes[0].plot(self.learning_curve)

        last_train_accuracy = self.train_accuracy_curve[-1]
        last_val_accuracy = self.val_accuracy_curve[-1]
        best_val_accuracy = self.best_val_accuracy
        best_val_f1 = self.best_val_f1
        
        axes[1].set_title(f'Train {last_train_accuracy:.4f}, val {last_val_accuracy:.4f}, '
                          f'max val {self.best_val_accuracy:.4f} at {self.best_val_epoch}, '
                          f'max f1 {self.best_val_f1:.4f} at {self.best_val_f1_epoch}')
        axes[1].plot(self.train_accuracy_curve)
        axes[1].plot(self.val_accuracy_curve)

        plt.tight_layout()
        plt.show()


def set_random_seeds(seed_value=0, device='cpu'):
    '''source https://forums.fast.ai/t/solved-reproducibility-where-is-the-randomness-coming-in/31628/5'''
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu':
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False