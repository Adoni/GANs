
# coding: utf-8

# In[4]:


import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
import sys


# In[52]:


import matplotlib
import numpy
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from torchvision import utils
show_image=False
def imshow(inp, file_name, save=False, title=None):
    """Imshow for Tensor."""
    fig = plt.figure(figsize=(10, 10))
    inp = inp.numpy().transpose((1, 2, 0))
#     inp-=numpy.min(inp)
    plt.imshow(inp)
    plt.savefig(file_name)
    if show_image:
        plt.show()
    else:
        print("Don't show")


# In[6]:


z_size=128
hidden_size=64
img_size=32


# In[7]:


root = './data'
download = True
trans = transforms.Compose([
    transforms.Scale(img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_set = dset.MNIST(
    root=root, train=True, transform=trans, download=download)
test_set = dset.MNIST(root=root, train=False, transform=trans)
batch_size = 128
kwargs = {}
train_loader = torch.utils.data.DataLoader(
    dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    dataset=test_set, batch_size=batch_size, shuffle=False)


# In[45]:


class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(z_size,hidden_size*8,4,2,1),
            nn.BatchNorm2d(hidden_size*8),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_size*8,hidden_size*4,4,2,1),
            nn.BatchNorm2d(hidden_size*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_size*4,hidden_size*2,4,2,1),
            nn.BatchNorm2d(hidden_size*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_size*2,hidden_size,4,2,1),
            nn.BatchNorm2d(hidden_size*1),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_size,1,4,2,1),
            nn.Tanh(),
        )
    def forward(self, x):
        x = x.view(x.size()[0], z_size, 1, 1)
        out = self.model(x)
        out = out.view(x.size()[0], 1, img_size,img_size)
        return out


# In[46]:


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1,hidden_size,4,2,1),
            nn.BatchNorm2d(hidden_size),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(hidden_size,hidden_size*2,4,2,1),
            nn.BatchNorm2d(hidden_size*2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(hidden_size*2,hidden_size*4,4,2,1),
            nn.BatchNorm2d(hidden_size*4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(hidden_size*4,hidden_size*8,4,2,1),
        )
        self.linear=nn.Sequential(
            nn.Linear(hidden_size*8*2*2,1)
        )
    def forward(self, x):
        out = self.model(x.view(x.size(0), 1, img_size, img_size))
        out=out.view(x.size()[0],hidden_size*8*2*2)
        out=self.linear(out)
        out=out.mean(0)
        out = out.view(1)
        return out


# In[47]:


one = torch.FloatTensor([1])
noise_holder=torch.FloatTensor(batch_size, z_size, 1, 1)
input_holder = torch.FloatTensor(batch_size, 1, img_size, img_size)
if torch.cuda.is_available():
    one=one.cuda()
    noise_holder=noise_holder.cuda()
    input_holder=input_holder.cuda()
mone = one * -1


# In[48]:


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# In[51]:


from tqdm import tqdm
G = Generator()
G.apply(weights_init)
D = Discriminator()
print(G)
if torch.cuda.is_available():
    G.cuda()
    D.cuda()
G_lr = D_lr = 5e-5
optimizers = {
    'D': torch.optim.RMSprop(D.parameters(), lr=D_lr),
    'G': torch.optim.RMSprop(G.parameters(), lr=G_lr)
}
criterion = nn.BCELoss()
err_Ds=[]
for epoch in tqdm(range(10000)):
    for p in D.parameters():
        p.requires_grad = True
    if epoch<25 or epoch%100==0:
        iter_D=1
    else:
        iter_D=1
    for _ in range(iter_D):
        for p in D.parameters():
            p.data.clamp_(-0.01, 0.01)
        optimizers['D'].zero_grad()
        data=next(iter(train_loader))[0]
        if torch.cuda.is_available():
            data=data.cuda()
        input_holder.resize_as_(data).copy_(data)
        output_real = D(Variable(data))
        output_real.backward(one)
        noise_holder.resize_(data.size()[0], z_size, 1, 1).normal_(0, 1)
        noisev = Variable(noise_holder,volatile=True)
        fake_data = Variable(G(noisev).data)
        output_fake = D(fake_data)
        output_fake.backward(mone)
        err_Ds.append((output_real-output_fake).data.numpy()[0])
        optimizers['D'].step()

    for p in D.parameters():
        p.requires_grad = False
    optimizers['G'].zero_grad()
    noise_holder.resize_(data.size()[0], z_size, 1, 1).normal_(0, 1)
    noisev = Variable(noise_holder)
    fake_data = G(noisev)
    output_fake1 = D(fake_data)
    output_fake1.backward(one)
#     G_loss = -torch.mean(output_fake1)

#     G_loss.backward()
    optimizers['G'].step()

    if epoch % 100 == 0:
        if torch.cuda.is_available():
            dd = utils.make_grid(fake_data.cpu().data[:64])
        else:
            dd = utils.make_grid(fake_data.data[:64])
        imshow(dd,'./results/WGAN_DC_%d.png'%(epoch))
        dd = dd.mul(0.5).add(0.5)
        vutils.save_image(dd, './results/OrWGAN_DC_%d.png'%(epoch))
        #plt.plot(err_Ds)
        #plt.show()


# In[60]:


import torchvision.utils as vutils
data=next(iter(train_loader))[0]
data = data.mul(0.5).add(0.5)
vutils.save_image(data, 'my.png')


# In[72]:


err_Ds[0].data.numpy()[0]

