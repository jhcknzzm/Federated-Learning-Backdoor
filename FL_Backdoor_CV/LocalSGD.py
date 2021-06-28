import torch
import torch.distributed as dist
from torch.optim.optimizer import Optimizer, required
from comm_helpers import communicate, flatten_tensors, unflatten_tensors
import threading


class SGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v

        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v = \rho * v + lr * g \\
             p = p - v

        The Nesterov version is analogously modified.
    """

    def __init__(self, params, gmf, tau, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, variance=0):

        self.gmf = gmf
        # self.size = size
        self.comm_buf = []
        self.itr = 0
        self.cp = tau


        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, variance=variance)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)


        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                buf = param_state['anchor_model'] = torch.clone(p.data).detach()
                self.comm_buf.append(buf)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']


            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]

                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss

    # def average(self):
    #     step_flag = (self.itr != 0 and self.itr % self.cp == 0)
    #     self.itr += 1
    #     if step_flag:
    #         if self.gmf == 0:
    #             # simple average
    #             param_list = []
    #             for group in self.param_groups:
    #                 for p in group['params']:
    #                     p.data.div_(self.size)
    #                     param_list.append(p.data)
    #             communicate(param_list, dist.all_reduce)
    #         else:
    #             # simple average + global momentum
    #             for group in self.param_groups:
    #                 lr = group['lr']
    #                 for p in group['params']:
    #                     param_state = self.state[p]
    #                     old_data = param_state['anchor_model']
    #
    #                     if 'global_momentum_buffer' not in param_state:
    #                         buf = param_state['global_momentum_buffer'] = torch.clone(p.data).detach()
    #                         buf.sub_(old_data)
    #                         buf.div_(-lr)
    #                     else:
    #                         buf = param_state['global_momentum_buffer']
    #                         buf.mul_(self.gmf).sub_(1/lr, p.data).add_(1/lr, old_data)
    #
    #                     old_data.add_(-lr, buf)
    #                     old_data.div_(self.size)
    #
    #             communicate(self.comm_buf, dist.all_reduce)
    #             for group in self.param_groups:
    #                 for p in group['params']:
    #                     param_state = self.state[p]
    #                     old_data = param_state['anchor_model']
    #                     p.data.copy_(old_data)
