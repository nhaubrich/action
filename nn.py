import torch
from torch import nn

#torch.manual_seed(0)
import seaborn as sns
import matplotlib.pyplot as plt
import pdb

#TODO
#Find good number of points
#Improve derivate stability
#split layers in half for ReLU/IReLU?

class IReLU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return (x>0)*0.5*x**2

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
                #nn.Linear(N,N//2),
                #IReLU(),
                #nn.Linear(N//2,N//4),
                #nn.ReLU(),
                nn.Linear(N//1,1)
        )
        #doesn't seem better. hybrid relu, x:x for 0-1, then x^2/2 above?
        self.basestack = nn.Sequential(
                nn.Linear(1,N)
        )
        self.relu = nn.Sequential(
                nn.ReLU(),
                nn.Linear(N//2,N//2),
                IReLU(),
        )
        self.irelu = nn.Sequential(
                IReLU(),
                nn.Linear(N//2,N//2),
                nn.ReLU(),
        )
        self.final = nn.Sequential(
                nn.Linear(N,1)
        )

    def forward(self,x):
        return self.stack(x)


NN = NeuralNetwork()


t0=0
tf=2
q0=0
qf=0
#qdot0=0
m=2
g=10

v0=(qf-q0)/(tf-t0)+g/2*(tf-t0)
print("v0",v0)
def Lagrangian(q,qdot,m,g):
    return 1/2*m*qdot**2 - m*g*q
    #return 1/2*m*qdot**2 - 1/2*m*q**2

def Hamiltonian(q,qdot,m,g):
    return 1/2*m*qdot**2 + m*g*q
    #return 1/2*m*qdot**2 + 1/2*m**q**2

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

epochs=2000
Npoints=2**10
NN.train()
try:
    for i in range(epochs):
        #avoid sorting by linear spacing + gaussian noise?
        #timesteps = torch.linspace(t0,tf,Npoints).reshape(-1,1)

        timesteps = torch.cat([torch.tensor(t0,dtype=torch.float).reshape(-1,1),
            torch.FloatTensor(Npoints-2,1).uniform_(t0,tf),
            torch.tensor(tf,dtype=torch.float).reshape(-1,1)])
        timesteps = timesteps.sort(dim=0)[0] #sorted for trap rule
        
        #timesteps.requires_grad = True
        
        NNqdot = NN(timesteps)
        NNq = torch.cat([ torch.zeros((1,1)),  torch.cumulative_trapezoid(NNqdot,x=timesteps,dim=0) ])

        NNq0 = NNq[0]
        NNqf = NNq[-1]
        #NNqdot0 = NNqdot[0]
        optimizer.zero_grad()
        
        C = ((qf-q0) - (NNqf-NNq0))/(tf-t0)

        #IC: add constant to NNqdot to match avg displacement
        qdot = NNqdot+C
        q = NNq+(C*timesteps-C*timesteps[0])#+q0


        L = Lagrangian(q,qdot,m,g)
        H = Hamiltonian(q,qdot,m,g)

        action = torch.trapezoid(L,x=timesteps,dim=0)
        BCloss = (torch.abs(NNqf-qf)+torch.abs(NNq0-q0)) 
        sigH = torch.std(H)
        loss =  action  #torch.dist(torch.abs(NNqf-NNq0),torch.tensor([1.0]))
        
        loss.backward()

        torch.nn.utils.clip_grad_norm(NN.parameters(),0.01)

        optimizer.step()
        scheduler.step(loss)
        optimizer.zero_grad()
        
        if i%1==0:
            print("{}\taction {:.4f},  lr {:.2E}".format(i,loss.item(),scheduler._last_lr[0]))
            

        losses.append(loss.item())

        #if i%(epochs//100)==0 or i+1==epochs:
        if (i%(100)==0 and i!=0) or i+1==epochs:
        #if  i+1==epochs or loss.item()>1e5:
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
            
except KeyboardInterrupt:
    pass

print("C: {:.3f}, D: {:.3f}".format(C.item(),q0))
#eval
fig, axs = plt.subplots(2,3)
axs[0,0].set_title("position")
for i,(t,path) in enumerate(zip(paths_t,paths_q)):
    #axs[0,0].plot(t,path,color="blue",alpha=(i/len(paths_t))**2)
    axs[0,0].plot(t,path,color="blue",alpha=0.1)

axs[0,0].plot(t,path,color="blue",alpha=1)
axs[0,0].plot(t,[-0.5*g*ts**2+v0*ts+q0 for ts in t],color="black",linestyle=(0,(5, 10)))



power=2
axs[0,1].set_title("velocity")
for i,(t,path) in enumerate(zip(paths_t,paths_qdot)):
    axs[0,1].plot(t,path,color="green",alpha=0.1)

axs[0,1].plot(t,path,color="green",alpha=1)
axs[0,1].plot(t,[-g*ts+v0 for ts in t],color="black",linestyle=(0,(5, 10)))

#check: integrate velocity
print("integrated velocity: {:.3f}".format(torch.trapezoid(torch.tensor(paths_qdot[-1]),x=torch.tensor(paths_t[-1]))))


axs[0,2].set_title("Lagrangian")
for i,(t,path) in enumerate(zip(paths_t,paths_L)):
    #axs[0,2].plot(t,path,color="red",alpha=(i/len(paths_t))**power)
    axs[0,2].plot(t,path,color="red",alpha=0.1)
axs[0,2].plot(t,path,color="red",alpha=1)
axs[0,2].plot(t,[m*(g*ts)**2+m/2*v0**2-2*m*v0*g*ts-m*g*q0 for ts in t],color="black",linestyle=(0,(5, 10)))

axs[1,0].set_title("NN q")
axs[1,1].set_title("NN qdot")
axs[1,2].set_title("action (training loss)")

for i,(t,path) in enumerate(zip(paths_t,paths_NNq)):
    axs[1,0].plot(t,path,color="purple",alpha=0.1)
axs[1,0].plot(t,path,color="purple",alpha=1)

for i,(t,path) in enumerate(zip(paths_t,paths_NNqdot)):
    axs[1,1].plot(t,path,color="orange",alpha=0.1)
axs[1,1].plot(t,path,color="orange",alpha=1)

axs[1,2].plot(losses)
fig.tight_layout()
plt.show()
