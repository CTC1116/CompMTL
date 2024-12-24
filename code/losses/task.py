import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union
import copy, random
import numpy as np
from dataset import transforms
import torchvision
from abc import abstractmethod
from scipy.optimize import minimize, Bounds, minimize_scalar

__all__ = ['SegCrossEntropyLoss', 'depthloss', 'normaloss', 'kdloss', 'noise_loss', 'noisy', 'mask', 'PCGrad', 'make_criterion', 'train_transforms', 'model_fit', 'FAMO', 'softkd_FAMO', 'cagrad_log', 'one_hotsoft_kd',
           'valid_transforms', 'DIST', 'Weight', 'freeze_layer', 'loss_factory', 'structure_loss', 'pixel_mask_kdloss', 'competition_loss', 'IMTL', 'cagrad', 'grad2vec', 'overwrite_grad', 'soft_kdgrad', 'soft_kdgrad_p',
           'soft_kd_w', 'cagrad_nyu', 'grad2vec_segformer','model_fit_c','overwrite_grad_segformer','KDAM', 'nll']

train_transforms = torchvision.transforms.Compose([ # from ATRC
    transforms.RandomScaling(scale_factors=[0.5, 2.0], discrete=False),
    transforms.RandomCrop(size=(448, 576), cat_max_ratio=0.75),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.PhotoMetricDistortion(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.PadImage(size=(448, 576)),
    transforms.AddIgnoreRegions(),
    transforms.ToTensor(),
])
# Testing 
valid_transforms = torchvision.transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.PadImage(size=(448, 576)),
    transforms.AddIgnoreRegions(),
    transforms.ToTensor(),
])

def model_fit(x_pred=None, x_output=None, task_type='semantic'):
    device = x_pred.device
    if len(x_output.shape) == 3:
        B, H, W = x_output.size()
    else:
        B, C, H, W = x_output.size()
    x_pred = F.interpolate(x_pred, (H, W), mode='bilinear', align_corners=True)

    # binary mark to mask out undefined pixel space
    binary_mask = (torch.sum(x_output, dim=1) != 0).float().unsqueeze(1).to(device)

    if task_type == 'semantic':
        # semantic loss: depth-wise cross entropy
        # loss = F.nll_loss(x_pred, x_output, ignore_index=-1)
        loss = nn.functional.cross_entropy(x_pred, x_output,ignore_index=-1)

    if task_type == 'depth':
        # depth loss: l1 norm
        loss = torch.sum(torch.abs(x_pred - x_output) * binary_mask) / torch.nonzero(binary_mask, as_tuple=False).size(0)

    if task_type == 'normal':
        # normal loss: dot product
        loss = 1 - torch.sum((x_pred * x_output) * binary_mask) / torch.nonzero(binary_mask, as_tuple=False).size(0)

    return loss

class model_fit_c(nn.Module):
    """
    Cross entropy loss with ignore regions.
    """
    def __init__(self,ignore_index=None,task_type='semantic', device=None):
        super(model_fit_c, self).__init__()
        if device != None:
            self.device = device
        else:
            self.device = None
        self.task_type = task_type
        self.ignore_index = ignore_index
    def forward(self, x_pred, x_output):
        if self.device==None:
            self.device = x_pred.device
        if len(x_output.shape) == 3:
            B, H, W = x_output.size()
        else:
            B, C, H, W = x_output.size()
        x_pred = F.interpolate(x_pred, (H, W), mode='bilinear', align_corners=True)

        # binary mark to mask out undefined pixel space
        binary_mask = (torch.sum(x_output, dim=1) != 0).float().unsqueeze(1).to(self.device)

        if self.task_type == 'semantic':
            # semantic loss: depth-wise cross entropy
            # loss = F.nll_loss(x_pred, x_output, ignore_index=-1)
            loss = nn.functional.cross_entropy(x_pred, x_output,ignore_index=-1)

        if self.task_type == 'depth':
            # depth loss: l1 norm
            loss = torch.sum(torch.abs(x_pred - x_output) * binary_mask) / torch.nonzero(binary_mask, as_tuple=False).size(0)

        if self.task_type == 'normal':
            # normal loss: dot product
            loss = 1 - torch.sum((x_pred * x_output) * binary_mask) / torch.nonzero(binary_mask, as_tuple=False).size(0)

        return loss

class SegCrossEntropyLoss(nn.Module):
    """
    Cross entropy loss with ignore regions.
    """
    def __init__(self, ignore_index=None, class_weight=None, balanced=False):
        super().__init__()
        self.ignore_index = ignore_index
        if balanced:
            assert class_weight is None
        self.balanced = balanced
        if class_weight is not None:
            self.register_buffer('class_weight', class_weight)
        else:
            self.class_weight = None

    def forward(self, out, label, reduction='mean'):
        label = torch.squeeze(label, dim=1).long()
        if len(label.shape) == 3:
            B, H, W = label.size()
        else:
            B, C, H, W = label.size()
        out = F.interpolate(out, (H, W), mode='bilinear', align_corners=True)
        if self.balanced:
            mask = (label != self.ignore_index)
            masked_label = torch.masked_select(label, mask)
            assert torch.max(masked_label) < 2  # binary
            num_labels_neg = torch.sum(1.0 - masked_label)
            num_total = torch.numel(masked_label)
            w_pos = num_labels_neg / num_total
            class_weight = torch.stack((1. - w_pos, w_pos), dim=0)
            loss = nn.functional.cross_entropy(
                out, label, weight=class_weight, ignore_index=self.ignore_index, reduction='none')
        else:
            loss = nn.functional.cross_entropy(out,
                                               label,
                                               weight=self.class_weight,
                                               ignore_index=self.ignore_index,
                                               reduction='none')
        if reduction == 'mean':
            n_valid = (label != self.ignore_index).sum()
            return (loss.sum() / max(n_valid, 1)).float()
        elif reduction == 'sum':
            return loss.sum()
        elif reduction == 'none':
            return loss
        
# class SegCrossEntropyLoss(nn.Module):
#     def __init__(self, ignore_index=-1, **kwargs):
#         super(SegCrossEntropyLoss, self).__init__()
#         self.task_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

#     def forward(self, inputs, targets):
#         if len(targets.shape) == 3:
#             B, H, W = targets.size()
#         else:
#             B, C, H, W = targets.size()
#             targets = torch.squeeze(targets, dim=1)
#         inputs = F.interpolate(inputs, (H, W), mode='bilinear', align_corners=True)
#         return self.task_loss(inputs, targets)

class depthloss(nn.Module):
    def __init__(self,ignore_index=None, device=None):
        super(depthloss, self).__init__()
        self.device = device
        self.ignore_index = ignore_index #255 for InvPT, 0 for CIRKD
        self.normalize = False

    # def forward(self, x_pred, x_output):
    #     B, C, H, W = x_output.size()
    #     x_pred = F.interpolate(x_pred, (H, W), mode='bilinear', align_corners=True)
    #     binary_mask = (torch.sum(x_output, dim=1) != 0).type(torch.FloatTensor).unsqueeze(1).to(self.device)
    #     loss = torch.sum(torch.abs(x_pred - x_output) * binary_mask) / torch.nonzero(binary_mask).size(0)
    #     return loss
    def forward(self, out, label, reduction='mean'):
        if self.normalize:
            out = nn.functional.normalize(out, p=2, dim=1)
        B, C, H, W =label.size()
        out = F.interpolate(out, (H, W), mode='bilinear', align_corners=True)

        mask = (label != self.ignore_index).all(dim=1, keepdim=True).to(self.device)
        n_valid = torch.sum(mask).item()
        masked_out = torch.masked_select(out, mask)
        masked_label = torch.masked_select(label, mask)
        if reduction == 'mean':
            return nn.functional.l1_loss(masked_out, masked_label, reduction='sum') / max(n_valid, 1)
        elif reduction == 'sum':
            return nn.functional.l1_loss(masked_out, masked_label, reduction='sum')
        elif reduction == 'none':
            return nn.functional.l1_loss(masked_out, masked_label, reduction='none')

class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()

    def forward(self, bottom):
        qn = torch.norm(bottom, p=2, dim=1).unsqueeze(dim=1) + 1e-12
        top = bottom.div(qn)

        return top

class nll(nn.Module):
    def __init__(self):
        super(nll, self).__init__()

    def forward(self, pred, gt, val=False):
        # print(pred.shape,gt.shape)
        if val:
            return F.nll_loss(pred, gt, size_average=False)
        else:
            return F.nll_loss(pred, gt)

class normaloss(nn.Module):
    def __init__(self,ignore_index, device):
        super(normaloss, self).__init__()
        self.device = device
        # self.normalize = Normalize()
        self.normalize = True
        self.loss_func = F.l1_loss
        self.size_average = True
        self.ignore_index = ignore_index
    # def forward(self, x_pred, x_output):
    #     B, C, H, W = x_output.size()
    #     x_pred = F.interpolate(x_pred, (H, W), mode='bilinear', align_corners=True)
    #     binary_mask = (torch.sum(x_output, dim=1) != 0).type(torch.FloatTensor).unsqueeze(1).to(self.device)
    #     loss = 1 - torch.sum((x_pred * x_output) * binary_mask) / torch.nonzero(binary_mask, as_tuple=False).size(0)
    #     return loss
    # def forward(self, out, label):
    #     assert not label.requires_grad
    #     B, C, H, W =label.size()
    #     out = F.interpolate(out, (H, W), mode='bilinear', align_corners=True)
    #     ignore_label=self.ignore_index
    #     mask = (label != ignore_label)
    #     n_valid = torch.sum(mask).item()

    #     if self.normalize is not None:
    #         out_norm = self.normalize(out)
    #         loss = self.loss_func(torch.masked_select(out_norm, mask), torch.masked_select(label, mask), reduction='sum')
    #     else:
    #         loss = self.loss_func(torch.masked_select(out, mask), torch.masked_select(label, mask), reduction='sum')

    #     if self.size_average:
    #         if ignore_label:
    #             ret_loss = torch.div(loss, max(n_valid, 1e-6))
    #             return ret_loss
    #         else:
    #             ret_loss = torch.div(loss, float(np.prod(label.size())))
    #             return ret_loss

    #     return loss
    def forward(self, out, label, reduction='mean'):
        if self.normalize:
            out = nn.functional.normalize(out, p=2, dim=1)
        B, C, H, W =label.size()
        out = F.interpolate(out, (H, W), mode='bilinear', align_corners=True)

        mask = (label != self.ignore_index).all(dim=1, keepdim=True).to(self.device)
        n_valid = torch.sum(mask).item()
        masked_out = torch.masked_select(out, mask)
        masked_label = torch.masked_select(label, mask)
        if reduction == 'mean':
            return nn.functional.l1_loss(masked_out, masked_label, reduction='sum') / max(n_valid, 1)
        elif reduction == 'sum':
            return nn.functional.l1_loss(masked_out, masked_label, reduction='sum')
        elif reduction == 'none':
            return nn.functional.l1_loss(masked_out, masked_label, reduction='none')


class kdloss(nn.Module):
    def __init__(self,weight=None):
        super(kdloss, self).__init__()
        self.weight = weight
    def forward(self, iters=None, s1_output=None, s2_output=None, s3_output=None, t1_output=None, t2_output=None, t3_output=None):
        loss, loss1, loss2, loss3 = 0,0,0,0
        if s1_output != None:
            if len(s1_output.shape) == 2:
                s1_output = s1_output / (s1_output.pow(2).sum(1) + 1e-6).sqrt().view(s1_output.size(0), 1)
                t1_output = t1_output / (t1_output.pow(2).sum(1) + 1e-6).sqrt().view(t1_output.size(0), 1)
            else:
                s1_output = s1_output / (s1_output.pow(2).sum(1) + 1e-6).sqrt().view(s1_output.size(0), 1, s1_output.size(2), s1_output.size(3))
                t1_output = t1_output / (t1_output.pow(2).sum(1) + 1e-6).sqrt().view(t1_output.size(0), 1, t1_output.size(2), t1_output.size(3))
            loss1 = (s1_output - t1_output).pow(2).sum(1).mean()
        if s2_output != None:
            if len(s2_output.shape) == 2:
                s2_output = s2_output / (s2_output.pow(2).sum(1) + 1e-6).sqrt().view(s2_output.size(0), 1)
                t2_output = t2_output / (t2_output.pow(2).sum(1) + 1e-6).sqrt().view(t2_output.size(0), 1)
            else:
                s2_output = s2_output / (s2_output.pow(2).sum(1) + 1e-6).sqrt().view(s2_output.size(0), 1, s2_output.size(2), s2_output.size(3))
                t2_output = t2_output / (t2_output.pow(2).sum(1) + 1e-6).sqrt().view(t2_output.size(0), 1, t2_output.size(2), t2_output.size(3))
            loss2 = (s2_output - t2_output).pow(2).sum(1).mean()
        if s3_output != None:
            s3_output = s3_output / (s3_output.pow(2).sum(1) + 1e-6).sqrt().view(s3_output.size(0), 1, s3_output.size(2), s3_output.size(3))
            t3_output = t3_output / (t3_output.pow(2).sum(1) + 1e-6).sqrt().view(t3_output.size(0), 1, t3_output.size(2), t3_output.size(3))
            loss3 = (s3_output - t3_output).pow(2).sum(1).mean()  

        loss = loss1*self.weight[0] + loss2*self.weight[1] + loss3*self.weight[2]
        return loss, [loss1,loss2,loss3]


def competition_loss(a_value=None, b_value=None, c_value=None, theta=1):
    loss = 0
    if c_value != None:
        loss += F.kl_div(a_value.log(), b_value, reduction='batchmean') * theta
        loss += F.kl_div(b_value.log(), c_value, reduction='batchmean') * theta
        loss += F.kl_div(a_value.log(), c_value, reduction='batchmean') * theta
    else:
        # loss = F.cross_entropy(a_value, b_value)*theta ## always been 0.5
        # print(a_value.log(), b_value)
        loss = F.kl_div(a_value.log(), b_value, reduction='batchmean') * theta
    
    return loss


class PCGrad():
    def __init__(self, optimizer, reduction='mean'):
        self._optim, self._reduction = optimizer, reduction
        return

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        '''
        clear the gradient of the parameters
        '''

        return self._optim.zero_grad(set_to_none=True)

    def step(self):
        '''
        update the parameters with the gradient
        '''

        return self._optim.step()

    def pc_backward(self, objectives):
        '''
        calculate the gradient of the parameters

        input:
        - objectives: a list of objectives
        '''

        grads, shapes, has_grads = self._pack_grad(objectives)
        pc_grad = self._project_conflicting(grads, has_grads)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)
        return

    def _project_conflicting(self, grads, has_grads, shapes=None):
        shared = torch.stack(has_grads).prod(0).bool()
        pc_grad, num_task = copy.deepcopy(grads), len(grads)
        for g_i in pc_grad:
            random.shuffle(grads)
            for g_j in grads:
                g_i_g_j = torch.dot(g_i, g_j)
                if g_i_g_j < 0:
                    g_i -= (g_i_g_j) * g_j / (g_j.norm()**2)
        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
        if self._reduction:
            merged_grad[shared] = torch.stack([g[shared]
                                           for g in pc_grad]).mean(dim=0)
        elif self._reduction == 'sum':
            merged_grad[shared] = torch.stack([g[shared]
                                           for g in pc_grad]).sum(dim=0)
        else: exit('invalid reduction method')

        merged_grad[~shared] = torch.stack([g[~shared]
                                            for g in pc_grad]).sum(dim=0)
        return merged_grad

    def _set_grad(self, grads):
        '''
        set the modified gradients to the network
        '''

        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                p.grad = grads[idx]
                idx += 1
        return

    def _pack_grad(self, objectives):
        '''
        pack the gradient of the parameters of the network for each objective
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grads, shapes, has_grads = [], [], []
        for obj in objectives:
            self._optim.zero_grad(set_to_none=True)
            obj.backward(retain_graph=True)
            grad, shape, has_grad = self._retrieve_grad()
            grads.append(self._flatten_grad(grad, shape))
            has_grads.append(self._flatten_grad(has_grad, shape))
            shapes.append(shape)
        return grads, shapes, has_grads

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self):
        '''
        get the gradient of the parameters of the network with specific 
        objective
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grad, shape, has_grad = [], [], []
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                # tackle the multi-head scenario
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p).to(p.device))
        return grad, shape, has_grad

class make_criterion(nn.Module):
    '''
    knowledge distillation loss
    '''
    def __init__(self, temperature=1):
        super(make_criterion, self).__init__()
        self.temperature = temperature

    def forward(self, pred, soft, temp=None):
        if temp != None:
            self.temperature = temp
        if len(soft.shape) == 4:
            B, C, h, w = soft.size()
            # if pred.ndim in [2,4]:
            scale_pred = pred.permute(0,2,3,1).contiguous().view(-1,C)
            scale_soft = soft.permute(0,2,3,1).contiguous().view(-1,C)
        else:
            B,C = soft.size()
            scale_pred, scale_soft = pred, soft
        if C != 1:
            p_s = F.log_softmax(scale_pred / self.temperature, dim=1)
            p_t = F.softmax(scale_soft / self.temperature, dim=1)
            loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.temperature**2)
        else:
            loss = F.mse_loss(scale_pred, scale_soft, reduction='mean')
            # p_s = torch.log(torch.sigmoid(scale_pred / self.temperature))
            # p_t = torch.sigmoid(scale_soft / self.temperature)
            # loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.temperature**2)
        return loss

def cagrad(grads, alpha=0.5, rescale=0):
    g1 = grads[:,0]
    g2 = grads[:,1]

    g11 = g1.dot(g1).item()
    g12 = g1.dot(g2).item()
    g22 = g2.dot(g2).item()

    g0_norm = 0.5 * np.sqrt(g11+g22+2*g12)

    # want to minimize g_w^Tg_0 + c*||g_0||*||g_w||
    coef = alpha * g0_norm
    def obj(x):
        # g_w = x*(g_1-g_2) + g_2
        # g_w^T g_0: x*0.5*(g11+g22-2g12)+(0.5+x)*(g12-g22)+g22
        # g_w^T g_w: x^2*(g11+g22-2g12)+2*x*(g12-g22)+g22
        return coef * np.sqrt(x**2*(g11+g22-2*g12)+2*x*(g12-g22)+g22+1e-8) + 0.5*x*(g11+g22-2*g12)+(0.5+x)*(g12-g22)+g22

    res = minimize_scalar(obj, bounds=(0,1), method='bounded')
    x = res.x

    gw_norm = np.sqrt(x**2*g11+(1-x)**2*g22+2*x*(1-x)*g12+1e-8)
    lmbda = coef / (gw_norm+1e-8)
    g = (0.5+lmbda*x) * g1 + (0.5+lmbda*(1-x)) * g2 # g0 + lmbda*gw
    if rescale== 0:
        return g
    elif rescale== 1:
        return g / (1+alpha**2)
    else:
        return g / (1 + alpha)


def soft_kd_w(grads, kd_grads):
    g1 = grads[:,0]
    gkd_0 = kd_grads[:,0]
    g11 = g1.dot(g1).item()
    gkd_01 = gkd_0.dot(g1).item()

    def obj(x):
        return -(x*g11+(1-x)*gkd_01)

    res = minimize_scalar(obj, bounds=(0,1), method='bounded')
    x = res.x
    g = x*g1+(1-x)*gkd_0

    return g, x

def cagrad_nyu(grads, alpha=0.5, rescale=0):
    GG = grads.t().mm(grads).cpu() # [num_tasks, num_tasks]
    g0_norm = (GG.mean()+1e-8).sqrt() # norm of the average gradient

    x_start = np.ones(3) / 3
    bnds = tuple((0,1) for x in x_start)
    cons=({'type':'eq','fun':lambda x:1-sum(x)})
    A = GG.numpy()
    b = x_start.copy()
    c = (alpha*g0_norm+1e-8).item()
    def objfn(x):
        return (x.reshape(1,3).dot(A).dot(b.reshape(3, 1)) + c * np.sqrt(x.reshape(1,3).dot(A).dot(x.reshape(3,1))+1e-8)).sum()
    res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
    w_cpu = res.x
    ww = torch.Tensor(w_cpu).to(grads.device)
    gw = (grads * ww.view(1, -1)).sum(1)
    gw_norm = gw.norm()
    lmbda = c / (gw_norm+1e-8)
    g = grads.mean(1) + lmbda * gw
    if rescale== 0:
        return g
    elif rescale == 1:
        return g / (1+alpha**2)
    else:
        return g / (1 + alpha)

    
def grad2vec(m, grads, grad_dims, task):
    # store the gradients
    grads[:, task].fill_(0.0)
    cnt = 0
    for p in m.pretrained.parameters():
            grad = p.grad
            if grad is not None:
                grad_cur = grad.data.detach().clone()
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[:cnt + 1])
                grads[beg:en, task].copy_(grad_cur.data.view(-1))
            cnt += 1
    # print(grads[:,task], len(grads[:,task])) #23508032
def grad2vec_segformer(m, grads, grad_dims, task):
    # store the gradients
    grads[:, task].fill_(0.0)
    cnt = 0
    for p in [p for name,p in m.named_parameters() if p.requires_grad and not any(excluded in name for excluded in ['linear_pred'])]:
            grad = p.grad
            if grad is not None:
                grad_cur = grad.data.detach().clone()
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[:cnt + 1])
                grads[beg:en, task].copy_(grad_cur.data.view(-1))
            cnt += 1
def overwrite_grad(m, newgrad, grad_dims, device):
    newgrad = newgrad * 2 # to match the sum loss
    cnt = 0
    for param in m.pretrained.parameters():
        beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
        en = sum(grad_dims[:cnt + 1])
        this_grad = newgrad[beg: en].contiguous().view(param.data.size())
        this_grad = this_grad.to(device)
        param.grad = this_grad.data.clone()
        cnt += 1
def overwrite_grad_segformer(m, newgrad, grad_dims, device):
    newgrad = newgrad * 2 # to match the sum loss
    cnt = 0
    # for param in m.pretrained.parameters():
    for param in [p for name,p in m.named_parameters() if p.requires_grad and not any(excluded in name for excluded in ['linear_pred'])]:
        beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
        en = sum(grad_dims[:cnt + 1])
        this_grad = newgrad[beg: en].contiguous().view(param.data.size())
        this_grad = this_grad.to(device)
        param.grad = this_grad.data.clone()
        cnt += 1


def cosine_similarity(x, y, eps=1e-8):
    return (x * y).sum(1) / (x.norm(dim=1) * y.norm(dim=1) + eps)
def pearson_correlation(x, y, eps=1e-8):
    # print(x.shape, y.shape) #semantic:torch.Size([7, 262144]) torch.Size([7, 262144]); 
    # print(x.mean(1).unsqueeze(1).shape) #semantic:[7,1], depth:[1,1]
    # ss
    if x.shape[1]==1:
        return cosine_similarity(x, y, eps)
    else:
        return cosine_similarity(x - x.mean(1).unsqueeze(1), y - y.mean(1).unsqueeze(1), eps)
def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()
def intra_class_relation(y_s, y_t):
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))

def cs_divergence(p, q):
    # Ensure the probability distributions sum to 1 and are positive
    p = p / p.sum(dim=1, keepdim=True)
    q = q / q.sum(dim=1, keepdim=True)
    
    # Calculate the dot products
    pq_dot_product = torch.sum(p * q, dim=1)
    p_dot_product = torch.sum(p * p, dim=1)
    q_dot_product = torch.sum(q * q, dim=1)
    
    # Calculate the Cauchy-Schwarz divergence
    cs_div = -torch.log(pq_dot_product / (torch.sqrt(p_dot_product * q_dot_product)+1e-8)+1e-8)
    
    # Mean over all samples
    cs_loss = torch.mean(cs_div)
    return cs_loss

class DIST(nn.Module):
    def __init__(self, beta=1., gamma=1.):
        super(DIST, self).__init__()
        self.beta = beta
        self.gamma = gamma

    def forward(self, y_s, y_t):
        assert y_s.ndim in (2, 4)
        num_classes = y_s.shape[1]
        if (y_s.ndim == 4) & (y_t.ndim==4):
            y_s = y_s.transpose(1, 3).reshape(-1, num_classes)
            y_t = y_t.transpose(1, 3).reshape(-1, num_classes)
        # elif (y_s.ndim == 4) & (y_t.ndim==3):
        #     B, H, W = y_t.shape
        #     y_s = y_s.transpose(1, 3).reshape(-1, num_classes)
        #     y_t_one_hot = torch.zeros(B * H * W, num_classes, device=y_t.device)
        #     y_t_one_hot.scatter_(1, y_t.view(-1, 1), 1)
        if num_classes!=1: #for depth task, output from softmax is 1
            y_s = y_s.softmax(dim=1) 
            y_t = y_t.softmax(dim=1)
        inter_loss = inter_class_relation(y_s, y_t)
        intra_loss = intra_class_relation(y_s, y_t)
        loss = self.beta * inter_loss + self.gamma * intra_loss
        return loss
    # def forward(self, y_s, y_t):
    #     assert y_s.ndim == 4 and y_t.ndim == 3
    #     num_classes = y_s.shape[1]
    #     



    #     if num_classes != 1:  # 对于非深度任务
    #         y_s = y_s.softmax(dim=1)
    #         y_t_one_hot = y_t_one_hot.softmax(dim=1)

    #     inter_loss = inter_class_relation(y_s, y_t_one_hot)
    #     intra_loss = intra_class_relation(y_s, y_t_one_hot)
    #     loss = self.beta * inter_loss + self.gamma * intra_loss
    #     return loss

class KDAM(nn.Module):
    def __init__(self, alpha=1., beta=1.):
        super(KDAM, self).__init__()
        self.beta = beta
        self.alpha = alpha

    def forward(self, y_s, y_t):
        assert y_s.ndim in (2, 4)
        if (y_s.ndim == 4) & (y_t.ndim==4):
            num_classes = y_s.shape[1]
            y_s = y_s.transpose(1, 3).reshape(-1, num_classes)
            y_t = y_t.transpose(1, 3).reshape(-1, num_classes)
        # elif (y_s.ndim == 4) & (y_t.ndim==3):
        #     B, H, W = y_t.shape
        #     y_s = y_s.transpose(1, 3).reshape(-1, num_classes)
        #     y_t_one_hot = torch.zeros(B * H * W, num_classes, device=y_t.device)
        #     y_t_one_hot.scatter_(1, y_t.view(-1, 1), 1)
        if num_classes!=1: #for depth task, output from softmax is 1
            y_s = y_s.softmax(dim=1)
            y_t = y_t.softmax(dim=1)
        inter_loss = cs_divergence(y_s, y_t)
        intra_loss = cs_divergence(y_s.transpose(0, 1), y_t.transpose(0, 1))
        loss = self.alpha * inter_loss + self.beta * intra_loss
        return loss

class Weight(torch.nn.Module):
    def __init__(self, task_len=None):
        super(Weight, self).__init__()
        if task_len == 2:
            self.weights = nn.Parameter(torch.FloatTensor([1.0, 1.0]))
        else:
            self.weights = nn.Parameter(torch.FloatTensor([1.0, 1.0, 1.0]))

class freeze_layer(torch.nn.Module):
    def __init__(self, model_parameters=None, ratio=None, end_ratio = None, max_iter = None):
        super(freeze_layer, self).__init__()
        self.init_ratio = ratio
        self.max_iter = max_iter
        self.end_ratio = end_ratio
        self.pre_param_lenth = len(model_parameters)

    def forward(self, model, I, iter,ratio=None, indices=None):
        if ratio == None:
            self.ratio = self.init_ratio - (self.init_ratio-self.end_ratio)*(iter/self.max_iter) #0.8->0.2
        else:
            self.ratio = ratio
        C_out = len(I)
        num_frozen = int(C_out * self.ratio)
        threshold = torch.kthvalue(I, num_frozen)[0] # k_th small data
        frozen_channels = I > threshold
        if indices is not None:
            frozen_channels_ = [0]*self.pre_param_lenth
            for i, (indice,frozen_channel) in enumerate(zip(indices, frozen_channels)):
                if frozen_channel==1:
                    frozen_channels_[indice] = 1
        else:
            frozen_channels_ = frozen_channels
        # for i, freeze in enumerate(frozen_channels):
        for i,(name, param) in enumerate(model.named_parameters()):
            if i >= len(frozen_channels_):
                break
            if frozen_channels_[i]==1:
                param.requires_grad = False

EPS = 1e-8 # for numerical stability


class WeightMethod:
    def __init__(self, n_tasks: int, device: torch.device, max_norm = 1.0):
        super().__init__()
        self.n_tasks = n_tasks
        self.device = device
        self.max_norm = max_norm

    @abstractmethod
    def get_weighted_loss(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor],
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ],
        last_shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor],
        representation: Union[torch.nn.parameter.Parameter, torch.Tensor],
        **kwargs,
    ):
        pass

    def backward(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        last_shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        representation: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, None], Union[dict, None]]:
        """

        Parameters
        ----------
        losses :
        shared_parameters :
        task_specific_parameters :
        last_shared_parameters : parameters of last shared layer/block
        representation : shared representation
        kwargs :

        Returns
        -------
        Loss, extra outputs
        """
        loss, extra_outputs = self.get_weighted_loss(
            losses=losses,
            shared_parameters=shared_parameters,
            task_specific_parameters=task_specific_parameters,
            last_shared_parameters=last_shared_parameters,
            representation=representation,
            **kwargs,
        )

        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)
        loss.backward()
        return loss, extra_outputs

    def __call__(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        **kwargs,
    ):
        return self.backward(
            losses=losses,
            shared_parameters=shared_parameters,
            task_specific_parameters=task_specific_parameters,
            **kwargs,
        )

    def parameters(self) -> List[torch.Tensor]:
        """return learnable parameters"""
        return []


######## Yun_Achievement-Based_Training_Progress_Balancing_for_Multi-Task_Learning_ICCV_2023 ########
from abc import abstractmethod
import math as m
from functools import partial

def softmax(weights):
    return torch.softmax(torch.stack([value for value in weights.values()]), dim=-1)


def normalize(weights: dict):
    norm_const = 1 / sum(weights.values())
    return [value * norm_const for value in weights.values()]


class AccuracyBase:
    def __init__(self, applications, weight, *args, **kwargs):
        super(AccuracyBase, self).__init__(applications, weight, *args, **kwargs)
        self.kpi = {app: float('nan') for app in applications}
        self.momentum = 0.1

    def update_kpi(self, results):
        kpi = self._update_kpi(results)
        self._update_weight(kpi)

    @staticmethod
    def _update_kpi(results):
        key_dict = {
            'detection': ['mAP', 'mAP_2007', 'mAP_2012'],
            'segmentation': ['mIoU'],
            'depth': ['d1'],
            'normal': ['11.25'],
        }

        kpi = {key: (lambda x: sum(x) / len(x) if len(x) else None)([results[k] for k in value if k in results])
               for key, value in key_dict.items()}
        return kpi

    @abstractmethod
    def _update_weight(self, kpi: dict):
        pass

class AMTL(AccuracyBase):
    """
    Achievement-based Multi-Task Loss
    task_weight = {(task_potential - current_accuracy) / task_potential} ** focusing_factor
                = (1 - current_accuracy / task_potential) ** focusing_factor
    """
    def __init__(self, application, weight=None, device=None, potential=None, focusing_factor=2, margin=0.05, normal_type='softmax'):
        assert len(application) == len(potential), f'{self.__class__.__name__} requires a task potential for each task'
        self.margin = margin
        self.focusing = focusing_factor
        self.device = device
        self.potential = {app: potential[i] * (1 + margin) if potential else 1.0
                          for i, app in enumerate(application)}
        self._weight_norm = softmax if normal_type == 'softmax' else normalize

        # Task weight initialization considering potential
        # weight = [(1 + margin - value) for key, value in self.potential.items()]
        super(AMTL, self).__init__(application, weight, device)

        normalized_weight = self._weight_norm(self.weight)
        self.weight = {key: value for key, value in zip(self.weight.keys(), normalized_weight)}
        self.len = len(self.weight)

    def get_configs(self):
        string = '\t\t- task potential: %s\n' % \
                  (', '.join(['%s - %.3f' % (key, value) for key, value in self.potential.items()]))
        string += '\t\t- focusing factor: %.2f\n' % self.focusing
        string += '\t\t- margin: %.2f\n' % self.margin
        string += '\t\t- init weight: %s\n' % \
                  ', '.join(['%s - %.3f' % (key, value) for key, value in self.weight.items()])
        return string

    def _update_weight(self, kpi):
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        for app, k in self.kpi.items():
            self.kpi[app] = kpi[app] if m.isnan(k) else self.momentum * kpi[app] + (1 - self.momentum) * k
            self.weight[app] = torch.tensor((1 - self.kpi[app] / self.potential[app]) ** self.focusing, device=self.device)
        normalized_weight = self._weight_norm(self.weight)
        self.weight = {key: value*self.len for key, value in zip(self.weight.keys(), normalized_weight)}
        print(self.weight)

class Losses:
    def __init__(self, application, weight=None, device=None):
        self.losses = dict()
        self.cur_loss = dict()
        self.device = device

        if weight is None:
            weight = [1 / len(application) for _ in range(len(application))]
        assert len(application) == len(weight), 'num of loss weights should be same with the num of applications'
        self.weight = {app: torch.tensor(weight[i], device=device if torch.cuda.is_available() else 'cpu')
                       for i, app in enumerate(application)}

    def register_loss(self, application, loss_type, **kwargs):
        task_loss_dict = {
            # 'detection': DetectionLoss,
            # 'segmentation': SegCrossEntropyLoss,
            # 'depth': depthloss,
            'segmentation': model_fit_c,
            'depth': model_fit_c,
            'normal': normaloss
        }
        assert application in task_loss_dict, '%s is not supported yet' % application
        self.losses[application] = task_loss_dict[application](**kwargs)
        self.kd_loss = make_criterion()

    def __call__(self, preds, gt=None, pseudo=None):
        if gt is None:
            gt = dict()
        if pseudo is None:
            pseudo = dict()

        loss = torch.zeros(1, device=self.device if torch.cuda.is_available() else 'cpu')
        for key, function in self.losses.items():
            # print(preds[key].shape, gt[key].shape, pseudo[key].shape)
            self.cur_loss[key] = function(preds[key], gt[key]) + self.kd_loss(preds[key], pseudo[key]) if key in gt or key in pseudo \
                else sum([pred.sum() for pred in preds[key]]) * 0
            loss += self.cur_loss[key] * self.weight[key].item()
        return loss

    def get_cur_loss(self):
        loss = torch.zeros(1, device=self.device if torch.cuda.is_available() else 'cpu')
        for key, function in self.losses.items():
            loss += self.cur_loss[key] * self.weight[key].item()
        return loss

    def __str__(self):
        report, loss_dict, total_loss = '', dict(), 0
        for app, loss in self.losses.items():
            loss_dict[app] = 0
            for name, value in loss.items():
                loss_dict[app] += value
            report += '%s: %3.3f ' % (app, loss_dict[app])
            total_loss += loss_dict[app] * self.weight[app]
        return report + 'total: %3.3f' % total_loss

    def items(self):
        total_loss, losses = 0, dict()
        for app, loss in self.losses.items():
            for key, value in loss.items():
                losses[key] = value
                total_loss += value * self.weight[app].item()
        losses['total_loss'] = total_loss
        return losses

    def get_weights(self):
        weights = dict()
        for app, weight in self.weight.items():
            weights[app+'_weight'] = weight.item()
        return weights

    def clear(self):
        for app, loss in self.losses.items():
            loss.clear()


class AMTLA(AMTL, Losses):
    """
    Achievement-based multi-task loss with Arithmetic mean
    """

    def get_configs(self):
        string = super(AMTLA, self).get_configs()
        string = '\t- Multi-Task Loss Metric: AMTL with Arithmetic Mean\n' + string
        return string


class WeightedSum(Losses):
    def __init__(self, application, weight=None):
        super(WeightedSum, self).__init__(application, weight)


loss_metric_list = [
    'weighted-sum',
    'geometric',
    'dwa', 'rlw',  # base on loss magnitude
    'grad-norm', 'imtl-g', 'imtl', 'mgda',  # based on gradient
    'dtp', 'fmtl', 'amtl', 'amtl-a',  # based on validation accuracy
    'pcgrad', 'pcgrad-amtl', 'cagrad', 'cagrad-amtl',  # based on optimization. independent to multi-task loss
    'kd-mtl', 'kd-amtl',  # based on knowledge distillation. independent to multi-task loss
    'dwa-g', 'rlw-g', 'dtp-g'  # Modified for geometric mean
]


def loss_factory(args, device):
    loss_metric = args.loss_metric.lower()

    loss_dict = {
        # 'amtl': partial(AMTLG, potential=args.potential, focusing_factor=args.focusing_factor, margin=args.margin),
        'amtl-a': partial(AMTLA, potential=args.potential, focusing_factor=args.focusing_factor, margin=args.margin),
        # 'kd-amtl': partial(AMTLG, potential=args.potential, focusing_factor=args.focusing_factor, margin=args.margin),
    }
    assert loss_metric in loss_dict, f'{loss_metric} is not supported for loss metric'
    loss_function = loss_dict[loss_metric](args.app, args.loss_weight, device=device)

    if 'detection' in args.app:
        loss_function.register_loss('detection', (args.cls_loss_type, args.loc_loss_type),
                                    prior=net.get_priors(), iou_aware=args.iou_aware_cls, mix_up=args.mix_up,
                                    label_smoothing=args.label_smoothing, loc_weight=args.loc_weight)
    if 'segmentation' in args.app:
        loss_function.register_loss('segmentation', 'cross_entropy', ignore_index=args.ignore_label,task_type='semantic')
    if 'depth' in args.app:
        loss_function.register_loss('depth', 'l1_loss', ignore_index=args.ignore_label, device=device,task_type='depth')
    if 'normal' in args.app:
        loss_function.register_loss('normal', 'cosine_loss', ignore_index=args.ignore_label, device=device)

    return loss_function
