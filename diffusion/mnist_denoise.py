import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np

EPOCHS = 10

train = datasets.MNIST("", train=True, download=False,
                        transform = transforms.Compose([transforms.ToTensor()]))

test = datasets.MNIST("", train=False, download=False,
                        transform = transforms.Compose([transforms.ToTensor()]))


batch_size = 10
train_set = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
test_set = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)


def noise(x, scale): return np.random.normal(scale=scale, size=x.shape)


def noise_batch(new_y, batch_size, old_y_data):
    new_x = []
    noised_batch = []
    for i in range(0, batch_size):
        noised_x = new_y[i] + noise(new_y[i], 0.1)
        noised_x = noised_x - noised_x.min()
        noised_x = noised_x/noised_x.max()
        noised_batch.append(noised_x.tolist())
        extra_row = [0] * 28
        extra_row[old_y_data[i]] = 1.0
        new_x.append(noised_x.view(-1).tolist() + extra_row)
        new_x_tensor = torch.tensor(new_x)
    
    new_batch = (new_x_tensor,new_y)

    return (torch.tensor(noised_batch), new_batch)

def data_noise_transform(data_set):
    new_dataset = []
    for data in data_set:
        old_y_data = data[1]
        
        new_y = data[0].clone().detach()
        noised_batch, new_batch = noise_batch(new_y, batch_size, old_y_data)
        new_dataset.append(new_batch)

        noised_batch, new_batch = noise_batch(noised_batch, batch_size, old_y_data)
        new_dataset.append(new_batch)

        noised_batch, new_batch = noise_batch(noised_batch, batch_size, old_y_data)
        new_dataset.append(new_batch)

        noised_batch, new_batch = noise_batch(noised_batch, batch_size, old_y_data)
        new_dataset.append(new_batch)


    return new_dataset

new_train_set = data_noise_transform(train_set)
new_test_set = data_noise_transform(test_set)

print("transformations complete")

# def accuracy():
#     correct = 0.0
#     total = 0.0
#     # lets test on test data
#     with torch.no_grad():
#         for data in train_set:
#             x, y = data
#             output = net(x.view(-1,28*28))
#             for idx, i in enumerate(output):
#                 if torch.argmax(i) == y[idx]:
#                     correct += 1
#                 total += 1

#     return round(correct/total, 3)

def visualize():
    # lets test on test data
    with torch.no_grad():
        img_to_show = 5
        idx = 0
        for data in new_train_set:
            x, y = data
            output = net(x.view(-1,29*28))
            plt.imshow(x[0].view(29,28))
            plt.show()
            
            plt.imshow(output[0].view(28,28))
            plt.show()
            
            plt.imshow(y[0].view(28,28))
            plt.show()
            
            idx += 1
            if idx > img_to_show:
                break
        plt.close()

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(812,1000)
        self.fc2 = nn.Linear(1000,784)
        # self.fc3 = nn.Linear(100,784)
        # self.fc3 = nn.Linear(784,784)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc3(x))

        return x


net = Net()
optimizer = optim.Adam(net.parameters(), lr = 0.001)

print(net)

for epoch in range(EPOCHS):
    print("Starting Epoch: ",epoch)
    for data in new_train_set:
        # data is batch
        x, y = data
        net.zero_grad()
        output = net(x.view(-1,29*28))
        # loss = F.nll_loss(output, y)
        loss = F.mse_loss(output, y.view(-1,28*28))
        loss.backward()
        optimizer.step()
    
    print("Loss: ", loss)
    if epoch >= 4: 
        visualize()
    # print("Accuracy: ", accuracy())
