import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

torch.manual_seed(0)



class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.img_width = 3840
        
        self.elinear1 = nn.Linear(3840, 1000)
        self.elinear2 = nn.Linear(1000, 1000)
        self.elinear3 = nn.Linear(1000, 100)
        self.eplinear1 = nn.Linear(100, latent_dim)
        self.eplinear2 = nn.Linear(100, latent_dim)
        
        self.dplinear1 = nn.Linear(latent_dim, 100) 
        self.dlinear1 = nn.Linear(100, 1000)
        self.dlinear2 = nn.Linear(1000, 1000)
        self.dlinear3 = nn.Linear(1000, 3840)
        
    def encode(self, x):
        x = self.elinear1(x)
        x = torch.relu(x)
        x = self.elinear2(x)
        x = torch.relu(x)
        x = self.elinear3(x)
        x = torch.relu(x)
        mean = self.eplinear1(x)
        logvar = self.eplinear2(x)
        return mean, logvar

    def decode(self, z):
        x = self.dplinear1(z)
        x = self.dlinear1(x)
        x = torch.relu(x)
        x = self.dlinear2(x)
        x = torch.relu(x)
        x = self.dlinear3(x)
        x = torch.relu(x)
        return x 
	
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar 


def vae_loss(recon_x, x, mu, logvar):
    mse = F.mse_loss(recon_x.reshape(-1, 3840), x.reshape(-1, 3840), reduction='none')
    kld = 0.5*(torch.sum(1 + logvar - mu**2 - torch.exp(logvar), axis=1))
    mse = torch.sum(mse, axis=1)
    mse = torch.mean(mse)
    kld = torch.mean(kld)
    #mse = F.mse_loss(recon_x.view(-1, img_width*img_height), x.view(-1, img_width*img_height))
    print ("LOSS = ", mse, kld)
    elbo = -mse+ kld 
    return -elbo 


#----------------------------------------------#
dataset = np.loadtxt("database.txt").astype(np.float32)
dataset = dataset.reshape(dataset.shape[0], dataset.shape[1])
#dataset = dataset[:50, :]
idx = int(0.8*len(dataset))
train_data = dataset[:idx, :]
test_data = dataset[idx:, :]
print (train_data.shape, test_data.shape)

batch_size = 32
train_dataset = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataset = DataLoader(test_data, batch_size=batch_size, shuffle=True)

#create model
latent_dim = 3 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

vae = VAE().to(device=device)
num_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
print('Number of parameters: %d' % num_params)
#vae.load_state_dict(torch.load("vae.pth"))

learning_rate = 1e-3
optimizer = torch.optim.Adam(params=vae.parameters(), lr=learning_rate)

EPOCHS = 1000 
print('Training ...')


for epoch in range(EPOCHS):
    vae.train()

    train_loss = 0 
    num_batches = 0
    
    for data in train_dataset:
        data = data.to(device)
        optimizer.zero_grad()
        # vae reconstruction
        data_recon, mu, logvar = vae(data)
        # reconstruction error
        loss = vae_loss(data_recon, data, mu, logvar)
        # backpropagation
        loss.backward()
        train_loss += loss.item()
        # one step of the optmizer (using the gradients from backpropagation)
        optimizer.step()
        num_batches += 1
        #print('Batch [%d / %d] ELBO loss: %f' % (num_batches, len(train_dataset), train_loss/num_batches))
        
    train_loss /= num_batches
    print('Epoch [%d / %d] ELBO loss: %f' % (epoch+1, EPOCHS, train_loss))

    torch.save(vae.state_dict(), "vae.pth")
    #np.savetxt("loss.txt", np.array(train_loss_avg))
vae = VAE()
vae.load_state_dict(torch.load("vae.pth"))

for i in range(10):
    np.random.seed(10000+i)
    z = np.random.normal(size=latent_dim).astype('float32')
    z = z.reshape((1, -1))
    z = torch.from_numpy(z)
    output = vae.decode(z)
    output_np = output.detach().numpy()[0]
    print (output_np.shape)
    plt.plot(output_np)
    filename = "test" + str(i) + ".png"
    plt.savefig(filename)
    plt.close()
