import numpy as np
from torch.autograd import Variable
import torch as torch
import copy
from torch.autograd.gradcheck import zero_gradients
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import torch as torch
import copy
from torch.autograd.gradcheck import zero_gradients
import time
import torch.nn as nn

class LabelSmoothLoss(nn.Module):
    def __init__(self, smoothing=0.9):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = 0.5

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * \
            self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss

def deepfool(image, net, num_classes = 10, overshoot = 0.2, max_iter = 50, attack_target=1):
    """
    :param image: Image of size 3*H*W
    :param net: network (input: images, output: values of activation **BEFORE** softmax).
    :param num_class:
    :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
    :param max_iter:
    :return:minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """
    # net.zero_grad()
    criterion_labelsmoothLoss = LabelSmoothLoss()
    f_image = net(Variable(image, requires_grad = True)).data.cpu().numpy().flatten()
    I = f_image.argsort()[::-1] #
    I = I[0:num_classes] #
    label = I[0] #

    input_shape = image.detach().cpu().numpy().shape #
    pert_image = copy.deepcopy(image) #
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    x = Variable(pert_image, requires_grad = True)
    target_class = attack_target

    fs = net(x)

    targets_u = np.zeros((fs.size(0))) + target_class

    targets_u = torch.from_numpy(targets_u).long()
    targets_u = targets_u.cuda()

    loss = (F.cross_entropy(fs, targets_u,
                          reduction='none') ).mean()

    k_i = label

    while k_i != target_class and loop_i < max_iter:

        zero_gradients(x) #
        loss.backward(retain_graph=True)

        cur_grad = x.grad.data.cpu().numpy().copy() #
        # set new w_k and new f_k
        w = cur_grad

        # compute r_i and r_tot
        ### (1e-1) need tuning
        r_i = (1e-1) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot - r_i) #r_total
        pert_image = image + (1+overshoot) * torch.from_numpy(r_tot).cuda()
        
        x = Variable(pert_image, requires_grad = True) #
        fs = net(x)
        k_i = np.argmax(fs.data.cpu().numpy().flatten()) #
        loop_i += 1

        fs = net(x)

        loss = (F.cross_entropy(fs, targets_u,
                              reduction='none') ).mean()

    r_tot = (1+overshoot)*r_tot

    return r_tot, loop_i


#### I have tried targetedfool(), but it works bad ...
def targetedfool(image, net, target,num_classes=10, overshoot=0.02, max_iter=1000):

    """
       :param image: Image of size HxWx3
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param target: target class label.
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """
    is_cuda = torch.cuda.is_available()

    if is_cuda:
        # print("Using GPU")
        image = image.cuda()
        net = net.cuda()
    else:
        pass
        # print("Using CPU")

    start = time.clock()

    f_image = net(Variable(image, requires_grad = True)).data.cpu().numpy().flatten()
    I = f_image.argsort()[::-1]
    I = I[0:num_classes]
    label = I[0] # original class label

    input_shape = image.cpu().detach().numpy().shape
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    x = Variable(pert_image, requires_grad = True)
    fs = net.forward(x)
    fs_list = [fs[0,I[k]] for k in range(num_classes)]
    k_i = label

    for t in I:
        if target == I[t]:
            break
        t = t + 1
    # print("t", t)

    while k_i != target and loop_i < max_iter:

        pert = np.inf
        fs[0, k_i].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()
        zero_gradients(x)

        fs[0, I[t]].backward(retain_graph=True)
        cur_grad = x.grad.data.cpu().numpy().copy()
        w_k = cur_grad - grad_orig
        f_k = (fs[0, I[t]] - fs[0, k_i]).data.cpu().numpy()

        pert_k = abs(f_k) / np.linalg.norm(w_k.flatten())

        # determine which w_k to use
        if pert_k < pert:
            pert = pert_k
            w = w_k

        r_i = (pert+1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)

        if is_cuda:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot).cuda()
        else:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot)

        x = Variable(pert_image, requires_grad=True)
        fs = net.forward(x)
        k_i = np.argmax(fs.data.cpu().numpy().flatten())

        loop_i += 1
        # print("lable print  ", loop_i, k_i)

    r_tot = (1+overshoot)*r_tot

    elapsed = (time.clock() - start)

    return r_tot, loop_i
