import torch
from torch import nn

#torch.manual_seed(0)
import matplotlib.pyplot as plt
import pdb

class IReLU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return (x>0)*0.5*x**2

class NeuralNetwork(nn.Module):
    def __init__(self):#,t0,tf,q0,qdot0):
        super().__init__()
        #self.t0=t0
        #self.tf=tf
        #self.q0=q0
        #self.qdot0=qdot0
        N = 64
        self.stack = nn.Sequential(
                nn.Linear(1,N),
                nn.ReLU(),
                nn.Linear(N,N),
                nn.ReLU(),
                nn.Linear(N,1,)
        )

    def forward(self,x):
        return self.stack(x)#-self.stack(torch.zeros(1))



NN = NeuralNetwork()


t0=0
tf=5
q0=0
qf=-8

#qdot0=0
m=2
g=10


def Lagrangian(q,qdot,m,g):
    return 1/2*m*qdot**2 - m*g*q

timesteps = torch.linspace(t0,tf,steps=11).reshape(-1,1)
timesteps.requires_grad = True
NNq = NN(timesteps)
NN(timesteps).backward(gradient=torch.ones(timesteps.shape))
NNqdot = timesteps.grad

#enforce I.C.
NNq0 = NNq[0]
NNqdot0 = NNqdot[0]
NNqf = NNq[-1]
#q = NNq-NNq0+q0 + (-NNqdot0+qdot0)*timesteps
#qdot = NNqdot-NNqdot0+qdot0
q = (NNq-NNq0)*(qf-q0)/(NNqf-NNq0)+q0
qdot = NNqdot*(qf-q0)/(NNqf-NNq0)


print("q0 {}, qf {}".format(q[0],q[-1]))

#training loop
#1. generate points
#2. compute action
#3. backprop
optimizer = torch.optim.Adam(NN.parameters(),lr=1e-4)
losses = []

paths_t = []
paths_q = []
paths_qdot = []
paths_L = []
paths_NNq = []
paths_NNqdot = []

epochs=800
Npoints=2**8
NN.train()
for i in range(epochs):
    #pdb.set_trace()
    #timesteps = torch.linspace(t0,tf,steps=1023).reshape(-1,1)
    
    timesteps = torch.cat([torch.tensor(t0,dtype=torch.float).reshape(-1,1),
        torch.FloatTensor(Npoints-2,1).uniform_(t0,tf),
        torch.tensor(tf,dtype=torch.float).reshape(-1,1)])
    timesteps.requires_grad = True
    
    NNq = NN(timesteps)
    NN(timesteps).backward(gradient=torch.ones(timesteps.shape))
    NNqdot = timesteps.grad

    NNq0 = NNq[0]
    NNqf = NNq[-1]
    #NNqdot0 = NNqdot[0]
    optimizer.zero_grad()

    q = (NNq-NNq0)*(qf-q0)/(NNqf-NNq0)+q0
    qdot = NNqdot*(qf-q0)/(NNqf-NNq0)
    #q = NNq-NNq0+q0 + (-NNqdot0+qdot0)*timesteps
    #qdot = NNqdot-NNqdot0+qdot0
    
    L = Lagrangian(q,qdot,m,g)
    action = L.mean()/(tf-t0)
    loss = action 

    #DEBUG loss = torch.sum((NNq-timesteps**2)**2)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print("{:.4f}".format(loss.item()))
    losses.append(loss.item())

    if i%(epochs//10)==0 or i+1==epochs:
        saved_t = timesteps.T.tolist()[0]
        saved_q = q.T.tolist()[0]
        saved_qdot = qdot.T.tolist()[0]
        saved_L = L.T.tolist()[0]
        saved_NNq = NNq.T.tolist()[0]
        saved_NNqdot = NNqdot.T.tolist()[0]

        saved_t,saved_q,saved_qdot,saved_L,saved_NNq,saved_NNqdot = zip(*sorted(zip(saved_t,saved_q,saved_qdot,saved_L,saved_NNq,saved_NNqdot)))
        paths_t.append(saved_t)
        paths_q.append(saved_q)
        paths_qdot.append(saved_qdot)
        paths_L.append(saved_L)
        paths_NNq.append(saved_NNq)
        paths_NNqdot.append(saved_NNqdot)

#eval

#NN.eval()
timesteps = torch.linspace(t0,tf,steps=1024).reshape(-1,1)
timesteps.requires_grad = True

NNq = NN(timesteps)
NN(timesteps).backward(gradient=torch.ones(timesteps.shape))
NNqdot = timesteps.grad

NNq0 = NNq[0]
NNqf = NNq[-1]
#NNqdot0 = NNqdot[0]

#q = NNq-NNq0+q0 + (-NNqdot0+qdot0)*timesteps
#qdot = NNqdot-NNqdot0+qdot0
q = (NNq-NNq0)*(qf-q0)/(NNqf-NNq0)+q0
qdot = NNqdot*(qf-q0)/(NNqf-NNq0)


L = Lagrangian(q,qdot,m,g)
action = L.mean()/(tf-t0)
#print("q0 {}, qdot0 {}".format(q[0],qdot[0]))

ts = timesteps.detach().numpy()
pos = q.detach().numpy()
vel = qdot.detach().numpy()

fig, axs = plt.subplots(2,3)

axs[0,0].set_title("position")
for i,(t,path) in enumerate(zip(paths_t,paths_q)):
    axs[0,0].plot(t,path,color="blue",alpha=i*1/len(paths_t))
#axs[0,0].plot(ts,-0.5*g*ts**2+qdot0*ts+q0,color="black",linestyle="--")

axs[0,1].set_title("velocity")
for i,(t,path) in enumerate(zip(paths_t,paths_qdot)):
    axs[0,1].plot(t,path,color="green",alpha=i*1/len(paths_t))
#axs[0,1].plot(ts,-g*ts+qdot0,color="black",linestyle="--")

axs[0,2].set_title("Lagrangian")
for i,(t,path) in enumerate(zip(paths_t,paths_L)):
    axs[0,2].plot(t,path,color="red",alpha=i*1/len(paths_t))
#axs[0,2].plot(ts,m*g**2*ts**2+m*qdot0**2/2,color="black",linestyle="--")

axs[1,0].set_title("NN q")
axs[1,1].set_title("NN qdot")
axs[1,2].set_title("action (training loss)")

for i,(t,path) in enumerate(zip(paths_t,paths_NNq)):
    axs[1,0].plot(t,path,color="purple",alpha=i*1/len(paths_t))

for i,(t,path) in enumerate(zip(paths_t,paths_NNqdot)):
    axs[1,1].plot(t,path,color="orange",alpha=i*1/len(paths_t))

#axs[1,0].plot(t,NNq.detach().numpy())
#axs[1,1].plot(t,NNqdot.detach().numpy())
axs[1,2].plot(losses)
fig.tight_layout()
plt.show()
