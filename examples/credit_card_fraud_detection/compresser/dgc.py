import torch
from torch.optim import SGD


class DGC_SGD(SGD):
    def __init__(self, params, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, sparsity=0.01, gradient_clip=0.01):
        super().__init__(params, lr, momentum, dampening, weight_decay, nesterov)
        self.sparsity = sparsity
        self.gradient_clip = gradient_clip

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                d_p.data.clamp_(-self.gradient_clip, self.gradient_clip)

                # Gradient Sparsification
                grad_abs = d_p.data.abs()
                mask = grad_abs.gt(self.sparsity * grad_abs.max())
                d_p.data.mul_(mask)

                # Momentum Correction
                if p in self.state and 'momentum_buffer' in self.state[p]:
                    buf = self.state[p]['momentum_buffer']
                    buf.mul_(mask)

                # Gradient Alignment
                if p.grad.data.norm() != 0:
                    p.grad.data /= p.grad.data.norm()

                if weight_decay != 0:
                    d_p.add_(p, alpha=weight_decay)

                if momentum != 0:
                    if 'momentum_buffer' not in self.state:
                        buf = self.state[p]['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = self.state[p]['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf
                p.add_(d_p, alpha=-group['lr'])
