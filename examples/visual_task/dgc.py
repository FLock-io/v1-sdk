import torch


def dgc(grads, sparsity: float = 0.9):
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

    return compressed_grads
