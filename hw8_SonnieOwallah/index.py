import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np 
import time

# loading MNIST dataset
image_path = './'
transform = transforms.Compose([transforms.ToTensor()])
mnist_dataset = torchvision.datasets.MNIST(
    root=image_path, train=True, transform=transform, download=True)

mnist_valid_dataset = Subset(mnist_dataset, torch.arange(10000))
mnist_train_dataset = Subset(mnist_dataset, torch.arange(10000, len(mnist_dataset)))
mnist_test_dataset = torchvision.datasets.MNIST(
    root=image_path, train=False, transform=transform, download=True)

print('number of items in mnist_dataset:', len(mnist_dataset))
print('number of items in mnist_valid_dataset:', len(mnist_valid_dataset))
print('number of items in mnist_train_dataset:', len(mnist_train_dataset))
print('number of items in mnist_test_dataset:', len(mnist_test_dataset))

# constructing data loader
batch_size = 64
torch.manual_seed(1)
train_dl = DataLoader(mnist_train_dataset, batch_size, shuffle=True)
valid_dl = DataLoader(mnist_valid_dataset, batch_size, shuffle=False)

# constructing a CNN in PyTorch
model = nn.Sequential(
    # layer 1: (28,28,1)
    nn.Conv2d(in_channels=1 , out_channels = 4, kernel_size= 3, stride=1, padding='valid'),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    # layer 2: (13,13,4)
    nn.Conv2d(in_channels=4 , out_channels = 2, kernel_size= 3, stride=3, padding='valid'),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=4, stride=4),
    # flatten: 2*1*1 = 2
    nn.Flatten(), 
    nn.Linear(2, 1024),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(1024, 10)
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# training model
def train(model, num_epochs, train_dl, valid_dl):
    loss_hist_train = [0] * num_epochs
    accuracy_hist_train = [0] * num_epochs
    loss_hist_valid = [0] * num_epochs
    accuracy_hist_valid = [0] * num_epochs

    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        model.train()
        for x_batch, y_batch in train_dl:
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            #calculate loss, accuracy
            loss_hist_train[epoch] += loss.item()*y_batch.size(0)
            is_correct = (
                torch.argmax(pred, dim=1) == y_batch
            ).float()
            accuracy_hist_train[epoch] += is_correct.sum()

        loss_hist_train[epoch] /= len(train_dl.dataset)    
        accuracy_hist_train[epoch] /= len(train_dl.dataset)

        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in valid_dl:
                pred = model(x_batch)
                loss = loss_fn(pred, y_batch)
                loss_hist_valid[epoch] +=  loss.item()*y_batch.size(0)
                is_correct = (
                    torch.argmax(pred, dim=1) == y_batch
                ).float()
                accuracy_hist_valid[epoch] += is_correct.sum()

            loss_hist_valid[epoch] /= len(valid_dl.dataset)    
            accuracy_hist_valid[epoch] /= len(valid_dl.dataset)

        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time

        print(f'Epoch {epoch+1} Accuracy: '
              f'{accuracy_hist_train[epoch]:.4f} Val_Accuracy: '
              f'{accuracy_hist_valid[epoch]:.4f} Epoch_Time: '
              f'{epoch_time:.4f}'
              )
    end_time = time.time()
    total_training_time = end_time - start_time
    print(f'Total Training Time: {total_training_time}')
    return loss_hist_train, loss_hist_valid, accuracy_hist_train, accuracy_hist_valid

# training model
torch.manual_seed(1)
num_epochs = 10
hist = train(model, num_epochs, train_dl, valid_dl)

# plotting training history
x_arr = np.arange(len(hist[0])) + 1
fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(1,2,1)
ax.plot(x_arr, hist[0], '-o', label = 'Train Loss')
ax.plot(x_arr, hist[1], '--<', label = 'Validation Loss')
ax.legend(fontsize=15)
ax.set_xlabel('Epoch', size=15)
ax.set_ylabel('Loss', size=15)

ax = fig.add_subplot(1,2,2)
ax.plot(x_arr, hist[2], '-o', label = 'Train Accuracy')
ax.plot(x_arr, hist[3], '--<', label = 'Validation Accuracy')
ax.legend(fontsize=15)
ax.set_xlabel('Epoch', size=15)
ax.set_ylabel('Accuracy', size=15)
plt.tight_layout()
plt.show()

with torch.no_grad():
    pred = model(mnist_test_dataset.data.unsqueeze(1).float() / 255.)
    is_correct = (
        torch.argmax(pred, dim=1) == mnist_test_dataset.targets
    ).float()
    print(f'Test Accuracy: {is_correct.mean():.4f}')


# visualizing predictions
fig = plt.figure(figsize=(12, 4))
for i in range(12):
    ax = fig.add_subplot(2, 6, i+1)
    ax.set_xticks([])
    ax.set_yticks([])
    img = mnist_test_dataset[i][0][0, :, :]
    with torch.no_grad():
        pred = model(img.unsqueeze(0). unsqueeze(0).float())
        y_pred = torch.argmax(pred)

    ax.imshow(img, cmap='gray_r')
    ax.text(0.9, 0.1, y_pred.item(),
            size = 15,
            color = 'blue',
            horizontalalignment = 'center',
            transform = ax.transAxes
            )
plt.show()
plt.close(fig)



           