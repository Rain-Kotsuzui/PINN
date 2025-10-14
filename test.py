import torch
x=torch.rand(2,2,requires_grad=True)
print(x)
y=x+2   
print(y)
z=torch.sum(y)
z.backward()
print(x.grad)