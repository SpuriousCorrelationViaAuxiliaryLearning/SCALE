import torch
from torch.optim import Optimizer
import matplotlib.pyplot as plt
import numpy as np


class AuxiliaryOptimizer(Optimizer):
    """Implements stochastic gradient descent (optionally with momentum).

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """

    def __init__(self, params, mode, group_size, lr=1e-3, momentum=0,
                 weight_decay=0):

        defaults = dict(lr=lr, momentum=momentum,
                        weight_decay=weight_decay)
        super(AuxiliaryOptimizer, self).__init__(params, defaults)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.group_size = group_size
        self.mode = mode
        self.reset()

    @torch.no_grad()
    def reset(self):
        self.gradients = {"train": None, "balanced": None}
        self.momentum_buffer = None
        self.params_with_grad_total = None

    @torch.no_grad()
    def store_gradients(self, name):
        params_with_grad_total = []
        d_p_list_total = []
        momentum_buffer_list_total = []
        for group in self.param_groups:
            # print(group)
            # Store all parameters with gradients, and their corresponding gradients
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            # Loop through each parameter
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(torch.clone(p.grad))
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            params_with_grad_total.append(params_with_grad)
            d_p_list_total.append(d_p_list)
            momentum_buffer_list_total.append(momentum_buffer_list)
        self.params_with_grad_total = params_with_grad_total
        self.momentum_buffer = momentum_buffer_list_total
        self.gradients[name] = d_p_list_total

    @torch.no_grad()
    def step(self):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        average_cosine_sim = []
        for params_with_grad, main_d_p_list, balanced_d_p_list, momentum_buffer_list in zip(self.params_with_grad_total, self.gradients["train"], self.gradients["balanced"], self.momentum_buffer):
            for i, param in enumerate(params_with_grad):
                main_d_p = main_d_p_list[i]
                balanced_d_p = balanced_d_p_list[i]

                # Group-wise adaptive rate
                # Every layer will have 4 adaptive learning rates
                # Bias and linear layers will only have one adaptive learning rate
                original_size = main_d_p.size()
                if len(original_size) > 2:
                    # For kernels, we group them into groups of kernels
                    #num_groups = int(main_d_p.size()[0] / self.group_size)
                    if main_d_p.size()[1] == 3:
                        num_groups = int(main_d_p.size()[0])
                    else:
                        num_groups = int(main_d_p.size()[0] * main_d_p.size()[1] / self.group_size)
                    main_d_p = main_d_p.view(num_groups, -1)
                    balanced_d_p = balanced_d_p.view(num_groups, -1)
                    adaptive_learning_rate = torch.nn.CosineSimilarity(dim=1)(torch.flatten(main_d_p, start_dim=1), torch.flatten(balanced_d_p, start_dim=1))
                    adaptive_learning_rate = adaptive_learning_rate.view(-1, 1)
                    cosine_sim = torch.flatten(adaptive_learning_rate).cpu().tolist()
                    average_cosine_sim.append(sum(cosine_sim)/len(cosine_sim))
                else:
                    # For biases, we simply flatten to get the cosine similarity
                    adaptive_learning_rate = torch.nn.CosineSimilarity(dim=0)(torch.flatten(main_d_p), torch.flatten(balanced_d_p))
                if self.mode == 0:
                    adaptive_learning_rate = (adaptive_learning_rate + 1) / 2
                elif self.mode == 1:
                    adaptive_learning_rate = torch.maximum(adaptive_learning_rate, torch.zeros_like(adaptive_learning_rate))
                elif self.mode == 2:
                    adaptive_learning_rate = adaptive_learning_rate
                elif self.mode == 3:
                    adaptive_learning_rate = torch.maximum(torch.sign(adaptive_learning_rate), torch.zeros_like(adaptive_learning_rate))
                    
                d_p = main_d_p * adaptive_learning_rate + balanced_d_p
                d_p = d_p.view(original_size)
                if self.weight_decay != 0:
                    d_p = d_p.add(param, alpha=self.weight_decay)

                if self.momentum != 0:
                    buf = momentum_buffer_list[i]

                    if buf is None:
                        buf = torch.clone(d_p).detach()
                        momentum_buffer_list[i] = buf
                    else:
                        buf.mul_(self.momentum).add_(d_p, alpha=1)

                    d_p = buf

                param.add_(d_p, alpha=-self.lr)

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer
        self.reset()
        '''
        plt.hist(average_cosine_sim, 100)
        plt.ylabel('Frequency')
        plt.xlabel('Cosine Similarity')
        plt.autoscale()
        plt.savefig(f"cosine_hist_{self.group_size}.png", format="png", dpi=1200)
        plt.clf()
        print(np.std(np.array(average_cosine_sim)))
        exit()
        '''
        print(average_cosine_sim)
        average_cosine_sim = sum(average_cosine_sim) / len(average_cosine_sim)
        return loss, average_cosine_sim
