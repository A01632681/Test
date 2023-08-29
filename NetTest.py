# import torch
# import tltorch
# import tensorly as tl
# import numpy as np
# import cv2
# import random
# import tensorly as tl
# from tensorly.decomposition import tucker

# # input_shape = (4, 5)
# # output_shape = (6, 2)
# # batch_size = 21
# # device = 'cpu'
# # x = torch.randn((batch_size,) + input_shape,
# #                 dtype=torch.float32, device=device)
# # print(x.shape)
# # trl = tltorch.TRL(input_shape, output_shape, rank='same')
# # result = trl(x)
# # print(result.shape)

# def read_frames(pathtoimages, max_frames):
#     frames = []
#     frame_count = 1
#     while True:
#         image = pathtoimages + 'Dep_' + str(frame_count) + '.png'
#         frame = cv2.imread(image)
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         print(frame.shape)
#         print(frame)
#         frame_norm = (frame - frame.min())/(frame.max() - frame.min())
#         print(frame_norm)
#         #frame = frame.reshape(1, 720, 1280)
#         frames.append(frame_norm)
#         frame_count += 1
#         if frame_count > max_frames:
#             break
#     return np.asarray(frames)

# path = r'D:\Documentos\Dissertation\CameraSamples\Path1\rec3\\'

# # Read in all the videos
# IMG_array = read_frames(path, max_frames=21)

# # Create tensors from matrices
# tensord = tl.tensor(IMG_array)

# random.seed(42) # Set the seed for reproducibility
# random_frames = random.sample(range(0, 21), 6) # Use these random frames to subset the tensors
# subset_lot = tensord[random_frames,:,:,:]
# subset_lot = subset_lot.type(torch.DoubleTensor) # Convert tensor to double

# core_lot, factors_lot = tucker(subset_lot, rank = [2,2,2,2]) # Perform Tucker decomposition
# print(factors_lot[0].shape)
# print(core_lot.shape)

# ------------------------------  
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy as np
import tensorly as tl
from tensorly.tenalg import inner
import tensorly as tl

tl.set_backend('pytorch')

batch_size = 16
# to run on CPU, uncomment the following line:
device = 'cpu'

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data/', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)

class TRL(nn.Module):
    def __init__(self, input_size, ranks, output_size, verbose=1, **kwargs):
        super(TRL, self).__init__(**kwargs)
        self.ranks = list(ranks)
        self.verbose = verbose

        if isinstance(output_size, int):
            self.input_size = [input_size]
        else:
            self.input_size = list(input_size)
           
        if isinstance(output_size, int):
            self.output_size = [output_size]
        else:
            self.output_size = list(output_size)
           
        self.n_outputs = int(np.prod(output_size[1:]))
       
        # Core of the regression tensor weights
        self.core = nn.Parameter(tl.zeros(self.ranks), requires_grad=True)
        self.bias = nn.Parameter(tl.zeros(1), requires_grad=True)
        weight_size = list(self.input_size[1:]) + list(self.output_size[1:])
       
        # Add and register the factors
        self.factors = []
        for index, (in_size, rank) in enumerate(zip(weight_size, ranks)):
            self.factors.append(nn.Parameter(tl.zeros((in_size, rank)), requires_grad=True))
            self.register_parameter('factor_{}'.format(index), self.factors[index])
       
        # FIX THIS
        self.core.data.uniform_(-0.1, 0.1)
        for f in self.factors:
            f.data.uniform_(-0.1, 0.1)

    def forward(self, x):
        regression_weights = tl.tucker_to_tensor((self.core, self.factors))
        return inner(x, regression_weights, n_modes=tl.ndim(x)-1) + self.bias
   
    def penalty(self, order=2):
        penalty = tl.norm(self.core, order)
        for f in self.factors:
            penatly = penalty + tl.norm(f, order)
        return penalty
   
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        self.trl = TRL(ranks=(10, 3, 3, 10), input_size=(batch_size, 50, 4, 4), output_size=(batch_size,10))

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = self.trl(x)
        return F.log_softmax(x)

model = Net()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion=nn.CrossEntropyLoss()
n_epoch = 20 # Number of epochs
regularizer = 0.001

model = model.to(device)

def train(n_epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
       
        # Important: do not forget to reset the gradients
        optimizer.zero_grad()
       
        output = model(data)
        loss = criterion(output,target) + regularizer*model.trl.penalty(2)
        loss.backward()
        optimizer.step()
        if batch_idx % 1000 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss = criterion(output,target)
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('mean: {}'.format(test_loss))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
       100. * correct / len(test_loader.dataset)))


for epoch in range(1, n_epoch):
    train(epoch)
    test()