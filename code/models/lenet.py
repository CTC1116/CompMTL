import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os

class MultiLeNetR(nn.Module):
    def __init__(self):
        super(MultiLeNetR, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc = nn.Linear(320, 50)
        
        self.task1_fc1 = nn.Linear(50, 50)
        self.task1_fc2 = nn.Linear(50, 10)
        self.task2_fc1 = nn.Linear(50, 50)
        self.task2_fc2 = nn.Linear(50, 10)

    def dropout2dwithmask(self, x, mask):
        channel_size = x.shape[1]
        if mask is None:
            mask = Variable(torch.bernoulli(torch.ones(1,channel_size,1,1)*0.5).to(x.device))
        mask = mask.expand(x.shape)
        return mask
    
    def zero_grad_shared_modules(self):
        for p in [p for name,p in self.named_parameters() if p.requires_grad and not any(excluded in name for excluded in ['task'])]:
            p.grad=None

    def forward(self, x, mask=None, mask_s=None):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.conv2(x)
        mask = self.dropout2dwithmask(x, mask)
        if self.training:
            x = x*mask
        x = F.relu(F.max_pool2d(x, 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc(x))
        mid_fea = x

        out1 = F.relu(self.task1_fc1(x))
        out2 = F.relu(self.task2_fc1(x))
        if mask_s is None:
            mask1 = Variable(torch.bernoulli(x.data.new(x.data.size()).fill_(0.5)))
            mask2 = Variable(torch.bernoulli(x.data.new(x.data.size()).fill_(0.5)))
        if self.training:
            out1 = out1*mask1
            out2 = out2*mask2
        out1 = self.task1_fc2(out1)
        out2 = self.task2_fc2(out2)

        return [F.log_softmax(out1, dim=1), F.log_softmax(out2, dim=1), mid_fea]

class LeNetR(nn.Module):
    def __init__(self):
        super(LeNetR, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc = nn.Linear(320, 50)
        
        self.task_fc1 = nn.Linear(50, 50)
        self.task_fc2 = nn.Linear(50, 10)

    def dropout2dwithmask(self, x, mask):
        channel_size = x.shape[1]
        if mask is None:
            mask = Variable(torch.bernoulli(torch.ones(1,channel_size,1,1)*0.5).to(x.device))
        mask = mask.expand(x.shape)
        return mask

    def forward(self, x, mask=None, mask_s=None):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.conv2(x)
        mask = self.dropout2dwithmask(x, mask)
        if self.training:
            x = x*mask
        
        x = F.relu(F.max_pool2d(x, 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc(x))
        mid_fea = x

        out = F.relu(self.task_fc1(x))
        if mask_s is None:
            mask_s = Variable(torch.bernoulli(x.data.new(x.data.size()).fill_(0.5)))
        if self.training:
            out = out*mask_s
        out = self.task_fc2(out)
        return [F.log_softmax(out, dim=1), mid_fea]

class MultiLeNetO(nn.Module):
    def __init__(self):
        super(MultiLeNetO, self).__init__()
        self.fc1 = nn.Linear(50, 50)
        self.fc2 = nn.Linear(50, 10)
    
    def forward(self, x, mask):
        x = F.relu(self.fc1(x))
        if mask is None:
            mask = Variable(torch.bernoulli(x.data.new(x.data.size()).fill_(0.5)))
        if self.training:
            x = x*mask
        x = self.fc2(x)
        return F.log_softmax(x, dim=1), mask

def get_lenet_multi(backbone='lenet', pretrained=None, img_size=[512,1024], num_class=40, task_len=None, norm_layer=nn.BatchNorm2d, **kwargs):
    if backbone == 'lenet':
        model = MultiLeNetR()

        if pretrained != 'None':
            #logger = get_root_logger()
            model.load_state_dict(torch.load(pretrained, map_location='cpu'))
            print('load pretrain.')
        return model
    else:
        raise ValueError('no such backbone')
    
    
def get_lenet(backbone='lenet', pretrained=None, img_size=[512,1024], num_class=40, task_len=None, norm_layer=nn.BatchNorm2d, **kwargs):
    if backbone == 'lenet':
        model = LeNetR()

        if pretrained != 'None':
            #logger = get_root_logger()
            model.load_state_dict(torch.load(pretrained, map_location='cpu'))
            print('load pretrain.')
        return model
    else:
        raise ValueError('no such backbone')