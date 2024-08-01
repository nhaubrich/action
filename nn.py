import torch
from torch import nn
import matplotlib.pyplot as plt
import pdb


class NeuralNetwork(nn.Module):
    def __init__(self):#,t0,tf,q0,qdot0):
        super().__init__()
        N = 64
        self.N = N
        self.stack = nn.Sequential(
                nn.Linear(1,N),
                nn.ReLU(),
                nn.Linear(N,N),
                nn.ReLU(),
                nn.Linear(N//1,1)
        )

    def forward(self,x):
        return self.stack(x)

NN = NeuralNetwork()

t0=0
tf=2
q0=0
qf=2
m=2
g=10



def Lagrangian(q,qdot,m,g):
    return 1/2*m*qdot**2 - m*g*q
    #return 1/2*m*qdot**2 - 1/2*m*q**2





#training 
optimizer = torch.optim.Adam(NN.parameters(),lr=1e-2)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4000, threshold=0.0001, threshold_mode='abs',eps=1e-15,cooldown=100)
losses = []

paths_t = []
paths_q = []
paths_qdot = []
paths_L = []
paths_NNq = []
paths_NNqdot = []

epochs=200
Npoints=2**10
NN.train()
try:
    for i in range(epochs):
        #timesteps = torch.linspace(t0,tf,Npoints).reshape(-1,1)

        timesteps = torch.cat([torch.tensor(t0,dtype=torch.float).reshape(-1,1),
            torch.FloatTensor(Npoints-2,1).uniform_(t0,tf),
            torch.tensor(tf,dtype=torch.float).reshape(-1,1)])
        timesteps = timesteps.sort(dim=0)[0] #sorted for trap rule
        
        NNqdot = NN(timesteps)
        NNq = torch.cat([ torch.zeros((1,1)),  torch.cumulative_trapezoid(NNqdot,x=timesteps,dim=0) ])

        NNq0 = NNq[0]
        NNqf = NNq[-1]
        optimizer.zero_grad()
        
        C = ((qf-q0) - (NNqf-NNq0))/(tf-t0)

        #IC: add constant to NNqdot to make qdot match avg displacement
        qdot = NNqdot+C
        q = NNq+(C*timesteps-C*timesteps[0])+q0

        L = Lagrangian(q,qdot,m,g)

        action = torch.trapezoid(L,x=timesteps,dim=0)
        BCloss = (torch.abs(NNqf-qf)+torch.abs(NNq0-q0)) 
        loss =  action
        
        loss.backward()

        optimizer.step()
        scheduler.step(loss)
        optimizer.zero_grad()
        
        if i%1==0:
            print("{}\taction {:.4f},  lr {:.2E}".format(i,loss.item(),scheduler._last_lr[0]))
            

        losses.append(loss.item())

        if (i%(10)==0 and i!=0) or i+1==epochs:
            paths_t.append( timesteps.T.tolist()[0])
            paths_q.append(q.T.tolist()[0])
            paths_qdot.append(qdot.T.tolist()[0])
            paths_L.append(L.T.tolist()[0])
            paths_NNq.append(NNq.T.tolist()[0])
            paths_NNqdot.append(NNqdot.T.tolist()[0])

            
except KeyboardInterrupt:
    pass

#eval
v0=(qf-q0)/(tf-t0)+g/2*(tf-t0)

fig, axs = plt.subplots(2,3)
plt.rcParams['figure.constrained_layout.use'] = True
axs[0,0].set_title("Position")
for i,(t,path) in enumerate(zip(paths_t,paths_q)):
    axs[0,0].plot(t,path,color="blue",alpha=0.1)

axs[0,0].plot(t,path,color="cyan",alpha=1,label="NN solution")
axs[0,0].plot(t,[-0.5*g*ts**2+v0*ts+q0 for ts in t],color="black",linestyle=(0,(5, 10)),label="Exact solution")
axs[0,0].set_xlabel("t")
axs[0,0].set_ylabel("x")
axs[0,0].legend()

axs[0,1].set_title("Velocity")
for i,(t,path) in enumerate(zip(paths_t,paths_qdot)):
    axs[0,1].plot(t,path,color="green",alpha=0.1)

axs[0,1].plot(t,path,color="limegreen",alpha=1,label="NN solution")
axs[0,1].plot(t,[-g*ts+v0 for ts in t],color="black",linestyle=(0,(5, 10)),label="Exact solution")
axs[0,1].set_xlabel("t")
axs[0,1].set_ylabel("v")
axs[0,1].legend()

axs[0,2].set_title("Lagrangian")
for i,(t,path) in enumerate(zip(paths_t,paths_L)):
    axs[0,2].plot(t,path,color="red",alpha=0.1)
axs[0,2].plot(t,path,color="red",alpha=1,label="NN solution")
axs[0,2].plot(t,[m*(g*ts)**2+m/2*v0**2-2*m*v0*g*ts-m*g*q0 for ts in t],color="black",linestyle=(0,(5, 10)),label="Exact solution")
axs[0,2].set_xlabel("t")
axs[0,2].legend()

axs[1,0].set_title("Integrated NN")
axs[1,1].set_title("NN")
axs[1,2].set_title("Action (Loss)")

for i,(t,path) in enumerate(zip(paths_t,paths_NNq)):
    axs[1,0].plot(t,path,color="purple",alpha=0.1)
axs[1,0].plot(t,path,color="purple",alpha=1)
axs[1,0].set_xlabel("t")

for i,(t,path) in enumerate(zip(paths_t,paths_NNqdot)):
    axs[1,1].plot(t,path,color="orange",alpha=0.1)
axs[1,1].plot(t,path,color="orange",alpha=1)
axs[1,1].set_xlabel("t")

axs[1,2].plot(losses)
plt.show()
