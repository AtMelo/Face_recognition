import torch.utils.data
from Data_Loader import CustomDataLoader
from Triplet_Loss import TripletLoss, get_triplet_loss
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from Conv_Network import ConvNetwork

embedding_dims = 128
batch_size = 4

transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
path_dir = r'C:\Users\toxa1\Desktop\Dataset Ex1\copy_files3'
data_set_img = CustomDataLoader(path_dir, transform)
train_loader = torch.utils.data.DataLoader(data_set_img, batch_size=batch_size, shuffle=True, drop_last=True)

model = ConvNetwork(embedding_dims)
criterion = TripletLoss()

def train(net, batch_size=256, epoch_num=50, epoch_info_show=10):
    optimizer = torch.optim.Adam(net.parameters(), lr=1.0e-3)
    for epoch in range(epoch_num):
        net.train()
        running_loss = []
        for anchor, positive, negative, anchor_label in train_loader:
            optimizer.zero_grad()
            anchor_out = net(anchor)
            positive_out = net(positive)
            negative_out = net(negative)

            loss = criterion(anchor_out, positive_out, negative_out)
            loss.backward()
            optimizer.step()

            running_loss.append(loss.detach().numpy())
            print("Epoch: {}/{} - Loss: {:.4f}".format(epoch + 1, epoch_num, np.mean(running_loss)))

train(model, batch_size, 1)
