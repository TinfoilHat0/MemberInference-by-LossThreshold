
import torch
import numpy as np
from torchvision import datasets, transforms




def get_datasets(data, augment=True):
    """ returns train and test datasets """
    train_dataset, test_dataset = None, None
    data_dir = '../data'
    
    if data == 'mnist':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.1307,), std=(0.3081,))])
        train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    elif data == 'fmnist':
        transform =  transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.2860], std=[0.3530])])
        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)   
    
    elif data == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform_test)
    
    #train_dataset.targets, test_dataset.targets = torch.FloatTensor(train_dataset.targets), torch.FloatTensor(test_dataset.targets)
    return train_dataset, test_dataset    



def get_loss_n_accuracy(model, data_loader, device='cuda:0', num_classes=10):
    """ Returns loss/acc, and per-class loss/accuracy on supplied data loader """
    
    with torch.inference_mode():
        # disable BN stats during inference
        model.eval()
        total_loss, correctly_labeled_samples = 0, 0
        confusion_matrix = torch.zeros(num_classes, num_classes)
        per_class_loss = torch.zeros(num_classes, device=device)
        per_class_ctr = torch.zeros(num_classes, device=device)
        
        criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)
        for _, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device=device, non_blocking=True),\
                    labels.to(device=device, non_blocking=True)
                                                        
            outputs = model(inputs)
            losses = criterion(outputs, labels)
            # keep track of total loss
            total_loss += losses.sum()
            # get num of correctly predicted inputs in the current batch
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correctly_labeled_samples += torch.sum(torch.eq(pred_labels, labels)).item()
            
            # per-class acc (filling confusion matrix)
            for t, p in zip(labels.view(-1), pred_labels.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            # per-class loss
            for i in range(num_classes):
                filt = labels == i
                per_class_loss[i] += losses[filt].sum()
                per_class_ctr[i] += filt.sum()
            
        loss = total_loss/len(data_loader.dataset)
        accuracy = correctly_labeled_samples / len(data_loader.dataset)
        per_class_accuracy = confusion_matrix.diag() / confusion_matrix.sum(1)
        per_class_loss = per_class_loss/per_class_ctr
        
        return (loss, accuracy), (per_class_accuracy, per_class_loss)

    


def get_loss_vals(model, data_loader, device):
    with torch.inference_mode():
        model.eval()
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        losses = torch.Tensor([]).to(device)
        
        for _, (inp, lbl) in enumerate(data_loader):
            inp, lbl = inp.to(device=device, non_blocking=True),\
                        lbl.to(device=device, non_blocking=True)
            
            out = model(inp)
            loss = criterion(out, lbl)
            losses = torch.cat([losses, loss])
    
    return losses
        
        

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    """ Taken from https://github.com/Bjarten/early-stopping-pytorch """
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        
    def __call__(self, val_loss, model=None):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            #self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            #self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss