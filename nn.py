import torch
from torch import nn

torch.manual_seed(0)

class NeuralNetwork(nn.Module):
    def __init__(self):#,t0,tf,q0,qdot0):
        super().__init__()
        #self.t0=t0
        #self.tf=tf
        #self.q0=q0
        #self.qdot0=qdot0

        self.stack = nn.Sequential(
                nn.Linear(1,64),
                nn.ReLU(),
                nn.Linear(64,64),
                nn.ReLU(),
                nn.Linear(64,1,)
        )

    def forward(self,x):
        return self.stack(x)#-self.stack(torch.zeros(1))

#class PIN(NeuralNetwork):
#    def __init__(self,t0,tf,q0,qdot0):
#        super().__init__()

#class Trajectory(nn.Module):
#    def __init__(self,t0,tf,q0,qdot0,NN):
#        super().__init__()
#        self.t0=t0
#        self.tf=tf
#        self.q0=q0
#        self.qdot0=qdot0
#        self.NN = NN
#
#    def forward(self,x):
#        self.NN(


NN = NeuralNetwork()


t0=0
tf=1
q0=0
qdot0=0

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
