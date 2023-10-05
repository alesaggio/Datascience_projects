import torch
import torch.nn as nn
import matplotlib.pylab as plt
import numpy as np
#torch.manual_seed(1)

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.data import DataLoader


######## Define some plotting functions ########

def plot_parameters(W, number_rows=1, name="", i=0):
    W = W.data[:, i, :, :]
    n_filters = W.shape[0]
    w_min = W.min().item()
    w_max = W.max().item()
    fig, axes = plt.subplots(number_rows, n_filters // number_rows)
    fig.subplots_adjust(hspace=0.4)

    for i, ax in enumerate(axes.flat):
        if i < n_filters:
            # Set the label for the sub-plot.
            ax.set_xlabel("kernel:{0}".format(i + 1))

            # Plot the image.
            ax.imshow(W[i, :], vmin=w_min, vmax=w_max, cmap='seismic')
            ax.set_xticks([])
            ax.set_yticks([])
    plt.suptitle(name, fontsize=10)    
    plt.savefig(f'params_{name}.png')

def plot_activations(A, number_rows=1, name="", i=0):
    A = A[0, :, :, :].detach().numpy()
    n_activations = A.shape[0]
    A_min = A.min().item()
    A_max = A.max().item()
    fig, axes = plt.subplots(number_rows, n_activations // number_rows)
    fig.subplots_adjust(hspace = 0.4)

    for i, ax in enumerate(axes.flat):
        if i < n_activations:
            # Set the label for the sub-plot.
            ax.set_xlabel("activation:{0}".format(i + 1))

            # Plot the image.
            ax.imshow(A[i, :], vmin=A_min, vmax=A_max, cmap='seismic')
            ax.set_xticks([])
            ax.set_yticks([])
    plt.savefig(f'activation_{name}.png')

def plot_loss_accuracy(loss_list, accuracy_list):
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.plot(loss_list, color=color)
    ax1.set_xlabel('epoch', color=color)
    ax1.set_ylabel('Loss', color=color)
    ax1.tick_params(axis='y', color=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.plot(accuracy_list, color=color)
    ax2.set_xlabel('epoch', color=color)
    ax2.set_ylabel('Accuracy', color=color)
    ax2.tick_params(axis='y', color=color)
    fig.tight_layout()
    plt.savefig("loss_accuracy.png")

def show_data(sample, shape = (28,28)):
    plt.imshow(sample[0].numpy().reshape(shape), cmap='gray')
    plt.title(f'label = {sample[1]}')
    plt.show()


# Download MNIST Fashion dataset and transform it into a tensor
train_dataset = datasets.FashionMNIST(root='./fashion_data', train=True, download=True, transform=transforms.ToTensor())
val_dataset =   datasets.FashionMNIST(root='./fashion_data', train=False, download=True, transform=transforms.ToTensor())

# Define module for Convolutional Neural Network
class myCNN(nn.Module):
    def __init__(self, ch_out1=8, ch_out2=12):
        super(myCNN, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=ch_out1, kernel_size=5, stride=1)
        #nn.init.kaiming_normal_(self.cnn1.weight)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        
        self.cnn2 = nn.Conv2d(in_channels=ch_out1, out_channels=ch_out2, kernel_size=3, stride=1)
        #nn.init.kaiming_normal_(self.cnn2.weight)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # fully-connected layer, dataset has 10 labels so 10 output nodes
        self.linear = nn.Linear(ch_out2*5*5, 10)

    def forward(self, x):
        x = self.cnn1(x)
        x = torch.relu(x)
        x = self.maxpool1(x)

        x = self.cnn2(x)
        x = torch.relu(x)
        x = self.maxpool2(x)

        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
    
    # Return outputs of all layers for debugging
    def activations(self, x):
        out1 = self.cnn1(x)
        out2 = torch.relu(out1)
        out3 = self.maxpool1(out2)
        out4 = self.cnn2(out3)
        out5 = torch.relu(out4)
        out6 = self.maxpool1(out5)
        return out1, out2, out3, out4, out5, out6

# Define model
model = myCNN(56, 112)

plot_parameters(model.state_dict()['cnn1.weight'], number_rows=8, name="cnn1_w_beforetraining")
plot_parameters(model.state_dict()['cnn2.weight'], number_rows=16, name="cnn2_w_beforetraining")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

train_loader = DataLoader(train_dataset, batch_size=100)
val_loader = DataLoader(val_dataset, batch_size=500)

loss_list = []
accuracy_list = []
N_val = len(val_dataset)
# Train the model
def train(epochs):
    for epoch in range(epochs):
        print(f'Training on epoch {epoch}')
        for x,y in train_loader:
            optimizer.zero_grad()
            z = model(x)
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
        loss_list.append(loss.item())

        #Evaluate accuracy on validation set
        correct = 0
        for x_, y_ in val_loader:
            z_ = model(x_)
            _, yhat = torch.max(z_.data, 1)
            correct += (yhat==y_).sum().item()
        accuracy = correct/N_val
        accuracy_list.append(accuracy)

epochs = 100
train(epochs)

plot_parameters(model.state_dict()['cnn1.weight'], number_rows=8, name="cnn1_w_aftertraining")
plot_parameters(model.state_dict()['cnn2.weight'], number_rows=16, name="cnn2_w_aftertraining")

out = model.activations(train_dataset[0][0].view(1, 1, 28, 28))

plot_activations(out[0], number_rows=4, name="cnn1_output")
plot_activations(out[1], number_rows=4, name="relu1_output")
plot_activations(out[3], number_rows=4, name="cnn2_output")
plot_activations(out[4], number_rows=4, name="relu2_output")

plot_loss_accuracy(loss_list, accuracy_list)

print('Final accuracy: ', accuracy_list[-1]*100)