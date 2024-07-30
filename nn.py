import torch
from torch import nn

#torch.manual_seed(0)
import matplotlib.pyplot as plt

class NeuralNetwork(nn.Module):
    def __init__(self):#,t0,tf,q0,qdot0):
        super().__init__()
        #self.t0=t0
        #self.tf=tf
        #self.q0=q0
        #self.qdot0=qdot0
        N = 4
        self.stack = nn.Sequential(
                nn.Linear(1,N),
                nn.ReLU(),
                nn.Linear(N,N),
                nn.ReLU(),
                nn.Linear(N,N),
                nn.ReLU(),
                nn.Linear(N,1,)
        )

    def forward(self,x):
        return self.stack(x)#-self.stack(torch.zeros(1))



NN = NeuralNetwork()


t0=0
tf=10
q0=0
qdot0=-0.1

timesteps=torch.tensor([t0,tf],dtype=torch.float,requires_grad=True)
f = timesteps**2
#f.backward(gradient=torch.tensor([1.0,1.0])

#zero = torch.zeros(1)
#zero.requires_grad = True
#NNq0 = NN(zero)
#NNq0.backward(gradient=torch.ones(1))
#NNqdot0 = zero.grad

#action


def Lagrangian(q,qdot):
    m=20
    g=10
    return 1/2*m*qdot**2 - m*g*q

timesteps = torch.linspace(t0,tf,steps=11).reshape(-1,1)
timesteps.requires_grad = True
NNq = NN(timesteps)
NN(timesteps).backward(gradient=torch.ones(timesteps.shape))
NNqdot = timesteps.grad

#enforce I.C.
NNq0 = NNq[0]
NNqdot0 = NNqdot[0]

q = NNq-NNq0+q0 + (-NNqdot0+qdot0)*timesteps
qdot = NNqdot-NNqdot0+qdot0


action = Lagrangian(q,qdot).sum()
print("q0 {}, qdot0 {}".format(q[0],qdot[0]))


#training loop
#1. generate points
#2. compute action
#3. backprop
optimizer = torch.optim.Adam(NN.parameters(),lr=1e-5)
losses = []
for i in range(1000):
    NN.train()
    #timesteps = torch.linspace(t0,tf,steps=1023).reshape(-1,1)
    
    timesteps = torch.cat([torch.tensor(t0,dtype=torch.float).reshape(-1,1),
        torch.FloatTensor(1023,1).uniform_(t0,tf)])
    timesteps.requires_grad = True
    
    NNq = NN(timesteps)
    NN(timesteps).backward(gradient=torch.ones(timesteps.shape))
    NNqdot = timesteps.grad

    NNq0 = NNq[0]
    NNqdot0 = NNqdot[0]

    q = NNq-NNq0+q0 + (-NNqdot0+qdot0)*timesteps
    qdot = NNqdot-NNqdot0+qdot0

    action = Lagrangian(q,qdot).sum()
    #print("q0 {}, qdot0 {}".format(q[0],qdot[0]))
    loss = action
    loss.backward()
    optimizer.step()
    print(q)
    optimizer.zero_grad()
    #print("zero opt")
    #print(qdot)
    print("{:.2f}".format(loss.item()))
    losses.append(loss.item())

#eval

#NN.eval()
timesteps = torch.linspace(t0,tf,steps=1024).reshape(-1,1)
timesteps.requires_grad = True

NNq = NN(timesteps)
NN(timesteps).backward(gradient=torch.ones(timesteps.shape))
NNqdot = timesteps.grad

NNq0 = NNq[0]
NNqdot0 = NNqdot[0]

q = NNq-NNq0+q0 + (-NNqdot0+qdot0)*timesteps
qdot = NNqdot-NNqdot0+qdot0

L = Lagrangian(q,qdot)
action = L.sum()
#print("q0 {}, qdot0 {}".format(q[0],qdot[0]))

t = timesteps.detach().numpy()
pos = q.detach().numpy()
vel = qdot.detach().numpy()

fig, axs = plt.subplots(4)
axs[0].set_title("position")
axs[1].set_title("velocity")
axs[2].set_title("instant action")
axs[3].set_title("training loss")
axs[0].plot(t,pos)
axs[1].plot(t,vel)
axs[2].plot(t,L.detach().numpy())
axs[3].plot(losses)
fig.tight_layout()
plt.show()
