import torch
from torch.optim import SGD


def dgc_func(grads, sparsity: float = 0.9):
    """
    This function implements the Deep Gradient Compression (DGC) for a given set of gradients.
    Args:
    grads: list of gradients to compress
    sparsity: the desired sparsity level, a float between 0 and 1.
    """
    # Flatten the gradients into a single 1-D tensor
    flat_grads = torch.cat([grad.view(-1) for grad in grads])

    # Compute the threshold
    abs_grads = flat_grads.abs()
    k = int(sparsity * flat_grads.numel())
    threshold = abs_grads.topk(k, largest=False).values.max()

    # Create a mask for the elements to keep
    mask = abs_grads.gt(threshold).float()

    # Apply the mask to the original gradients
    compressed_grads = []
    start = 0
    for grad in grads:
        end = start + grad.numel()
        compressed_grad = grad * mask[start:end].view_as(grad)
        compressed_grads.append(compressed_grad)
        start = end
    sparse_tensors = [compressed_grad.to_sparse() for compressed_grad in compressed_grads]

    return sparse_tensors


class DGC_SGD(SGD):
    def __init__(self, params, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, sparsity=0.01, gradient_clip=0.01,
                 store_compressed_grad=False):
        super().__init__(params, lr, momentum, dampening, weight_decay, nesterov)
        self.sparsity = sparsity
        self.gradient_clip = gradient_clip
        self.store_compressed_grad = store_compressed_grad
        self.compressed_grads = []

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        self.compressed_grads = []
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
                self.compressed_grads.append(d_p.clone())

                # Momentum Correction
                if p in self.state and 'momentum_buffer' in self.state[p]:
                    buf = self.state[p]['momentum_buffer']
                    buf.mul_(mask)

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
