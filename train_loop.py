import torch
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch.cuda.amp import GradScaler, autocast

plt.ion()


def train_loop(model, train_loader, valid_loader, test_loader, optimizer, criterion, epochs, device, path, save=False,
               scheduler=None, smart=True):
    stats = Stats(learning_rate=optimizer.param_groups[0]['lr'], epochs=epochs,
                  optimizer_name=optimizer.__class__.__name__, criterion_name=criterion.__class__.__name__)
    fig, ax = stats.live_plot()

    model = model.to(device)
    torch.cuda.empty_cache()

    epoch_without_changes = 0
    last_train_loss = 10
    early_stopping_after = 0
    early_stopping_loss_threshold = 0.0001

    for epoch in range(1, epochs + 1):
        early_stopping_after += 1
        train_losses, train_accuracy = train_one_epoch(model, train_loader, optimizer, criterion, device)
        valid_losses, valid_accuracy = validate(model, valid_loader, criterion, device)

        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_train_acc = sum(train_accuracy) / len(train_accuracy)
        avg_valid_loss = sum(valid_losses) / len(valid_losses)
        avg_valid_acc = sum(valid_accuracy) / len(valid_accuracy)

        if last_train_loss - avg_train_loss > early_stopping_loss_threshold:
            epoch_without_changes = 0
            last_train_loss = avg_train_loss
        else:
            epoch_without_changes += 1

        before_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            if smart and epoch_without_changes > 5:
                scheduler.step()
            if not smart:
                scheduler.step()
        after_lr = optimizer.param_groups[0]['lr']

        stats.update(avg_train_loss, avg_train_acc, avg_valid_loss, avg_valid_acc, after_lr, early_stopping_after)

        if epoch % 10 == 0 and save:
            stats.save_to_file(model, os.path.join(path, f'{epoch}'))
            stats.plot(os.path.join(path, f'{epoch}'))

        print(
            f'Epoch: {epoch:03d}, train loss: {avg_train_loss:.4f}, train accuracy: {avg_train_acc:.4f}, valid loss: {avg_valid_loss:.4f}, valid accuracy: {avg_valid_acc:.4f}, lr: {before_lr} -> {after_lr}, epoch without changes: {epoch_without_changes}')

        stats.update_plot(fig, ax)

        if smart and epoch_without_changes > 10:
            break

    print("Evaluating model on test set...")

    test_losses, test_accuracy = test(model, test_loader, criterion, device)
    torch.cuda.empty_cache()

    print(f'Test loss: {test_losses:.4f}, test accuracy: {test_accuracy:.4f}')

    stats.update_test(test_losses.float().item(), test_accuracy.float().item())
    stats.plot(path)

    if save:
        stats.save_to_file(model, os.path.join(path, 'last'))


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    scaler = GradScaler()
    train_losses = []
    train_accuracy = []
    model.train()
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        with autocast():
            output = model(data)
            loss = criterion(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_losses.append(loss.item())
        train_accuracy.append((output.argmax(1) == target).float().mean().item())

    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    return train_losses, train_accuracy


def validate(model, valid_loader, criterion, device):
    valid_losses = []
    valid_accuracy = []
    model.eval()

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(valid_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            valid_losses.append(loss.item())
            valid_accuracy.append((output.argmax(1) == target).float().mean().item())

    torch.cuda.empty_cache()
    return valid_losses, valid_accuracy


def test(model, test_loader, criterion, device):
    test_losses = []
    test_accuracy = []
    model.eval()

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(test_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_losses.append(loss.item())
            test_accuracy.append((output.argmax(1) == target).float().mean().item())

    torch.cuda.empty_cache()
    test_losses = torch.tensor(test_losses).mean(dtype=torch.float32).cpu()
    test_accuracy = torch.tensor(test_accuracy).mean(dtype=torch.float32).cpu()

    return test_losses, test_accuracy


class Stats:
    def __init__(self, learning_rate, epochs, optimizer_name, criterion_name):
        self.optimizer_name = optimizer_name
        self.criterion_name = criterion_name
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.train_losses = []
        self.train_accuracy = []
        self.valid_losses = []
        self.valid_accuracy = []
        self.test_losses = 0
        self.test_accuracy = 0
        self.learning_rates = []
        self.early_stopping_after = 0

    def update(self, train_losses, train_accuracy, valid_losses, valid_accuracy, lr, early_stopping_after):
        self.early_stopping_after = early_stopping_after
        self.learning_rates.append(lr)
        self.train_losses.append(train_losses)
        self.train_accuracy.append(train_accuracy)
        self.valid_losses.append(valid_losses)
        self.valid_accuracy.append(valid_accuracy)

    def update_test(self, test_losses, test_accuracy):
        self.test_losses = test_losses
        self.test_accuracy = test_accuracy

    def save_to_file(self, model, path):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(model.state_dict(), os.path.join(path, 'model.pt'))
        torch.save(self, os.path.join(path, 'stats.pt'))
        # save all data in json
        with open(os.path.join(path, 'stats.json'), 'w') as f:
            import json
            json.dump(self.__dict__, f)

    def plot(self, path):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Loss')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Accuracy')

        train_losses = torch.Tensor(self.train_losses).cpu()
        valid_losses = torch.Tensor(self.valid_losses).cpu()
        train_accuracy = torch.Tensor(self.train_accuracy).cpu()
        valid_accuracy = torch.Tensor(self.valid_accuracy).cpu()

        ax[0].plot(train_losses, label='train')
        ax[0].plot(valid_losses, label='valid')
        ax[0].set_title(
            f'{self.optimizer_name} - {self.criterion_name} - {self.epochs} epochs - lr: {self.learning_rate}')
        ax[0].legend()

        ax[1].plot(train_accuracy, label='train')
        ax[1].plot(valid_accuracy, label='valid')
        ax[1].set_title(
            f'{self.optimizer_name} - {self.criterion_name} - {self.epochs} epochs - lr: {self.learning_rate}')
        ax[1].legend()

        plt.savefig(os.path.join(path, 'training_progress.png'))

    def live_plot(self):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].set_title(
            f'{self.optimizer_name} - {self.criterion_name} - {self.epochs} epochs - lr: {self.learning_rate}')
        ax[1].set_title(
            f'{self.optimizer_name} - {self.criterion_name} - {self.epochs} epochs - lr: {self.learning_rate}')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Loss')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Accuracy')
        return fig, ax

    def update_plot(self, fig, ax):
        ax[0].clear()
        ax[1].clear()

        train_losses = torch.Tensor(self.train_losses).cpu()
        valid_losses = torch.Tensor(self.valid_losses).cpu()
        train_accuracy = torch.Tensor(self.train_accuracy).cpu()
        valid_accuracy = torch.Tensor(self.valid_accuracy).cpu()

        ax[0].plot(train_losses, label='train')
        ax[0].plot(valid_losses, label='valid')
        ax[0].legend()

        ax[1].plot(train_accuracy, label='train')
        ax[1].plot(valid_accuracy, label='valid')
        ax[1].legend()

        fig.canvas.draw()
        fig.canvas.flush_events()
