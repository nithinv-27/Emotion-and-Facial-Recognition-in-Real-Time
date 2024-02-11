import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import models
import torchvision.transforms as T
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

data_dir='./face_recognition'
img_size=[48,48]
batch_size=128

num_class=len(os.listdir(data_dir))
classes= os.listdir(data_dir)

train_ds=ImageFolder(data_dir, transform=T.Compose([
    T.Resize(img_size),
    T.ToTensor(),
]))

if __name__== "__main__":

    # print sample 
    img,label=train_ds[0]

    # load training data to a dataloader
    train_dl=DataLoader(train_ds, batch_size, shuffle=True, num_workers=2, pin_memory=True)

    # Define model 
    model = models.resnet18(weights='IMAGENET1K_V1')
    num_ftrs = model.fc.in_features  #num_ftrs=512
    model.fc=nn.Sequential(
        nn.Linear(num_ftrs, num_class),
        nn.Softmax(dim=1)
    ) #modifying fully connected layer to classify num_classes

    # Moving dataloader to device
    def get_default_device():
        """Pick GPU if available, else CPU"""
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    def to_device(data, device):
        """Move tensor(s) to chosen device"""
        if isinstance(data, (list,tuple)):
            return [to_device(x, device) for x in data]
        return data.to(device, non_blocking=True)

    class DeviceDataLoader():
        """Wrap a dataloader to move data to a device"""
        def __init__(self, dl, device):
            self.dl = dl
            self.device = device

        def __iter__(self):
            """Yield a batch of data after moving it to device"""
            for b in self.dl:
                yield to_device(b, self.device)

        def __len__(self):
            """Number of batches"""
            return len(self.dl)

    device=get_default_device() #get device
    print(device)

    train_dl=DeviceDataLoader(train_dl, device) #moving dataloader to device

    model=to_device(model, device) #moving model to device


    # Define loss function
    loss_fn=nn.CrossEntropyLoss()

    # Define optimizer
    optimizer=torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9,0.999))

    # Training loop
    def train_model(model, epochs, loss_fn, optimizer):
        losses=[]
        for epoch in range(epochs):
            with tqdm(train_dl, unit="batch") as tepoch:
                for img, label in tepoch:     

                    tepoch.set_description(f"Epoch {epoch}")      

                    # Make gradients=0
                    optimizer.zero_grad()

                    # Make predictions
                    output=model(img)
                    
                    # output=output.long()

                    # Calculate loss
                    loss=loss_fn(output, label)
                    print(type(loss))

                    # Calculate backward gradients
                    loss.backward()

                    # Adjust learning weights
                    optimizer.step()

                    # Calculate accuracy
                    preds=torch.argmax(output,dim=1)
                    accuracy=sum(preds==label)/batch_size

                    tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)

            losses.append(loss.item())
            
        return losses

    epochs=5
    history= train_model(model=model, epochs=epochs, loss_fn=loss_fn, optimizer=optimizer)

    loss_y=np.array(history)
    x_axis=np.arange(1,epochs+1)
    plt.xlabel('Total no. of epochs')
    plt.ylabel('Loss per epoch')
    plt.imshow(x_axis,loss_y)
    


        

        
    



