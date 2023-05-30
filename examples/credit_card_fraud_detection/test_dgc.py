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
    sparse_tensors = [compressed_grad.to_sparse() for compressed_grad in compressed_grads]
    return sparse_tensors


if __name__ == '__main__':
    grads = [torch.randn((10, 10)), torch.randn((20, 20)), torch.randn((30, 30))]

    compressed_grads = dgc(grads, sparsity=0.9)
    total_gradients = sum([grad.numel() for grad in grads])
    total_compressed_gradients = sum([compressed_grad.values().numel() for compressed_grad in compressed_grads])
    print(f'Compress rate. {(1 - total_compressed_gradients / total_gradients) * 100}')
