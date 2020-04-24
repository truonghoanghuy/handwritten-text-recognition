import numpy as np
import torch


def adjoint(a):
    """compute inverse without division by det; ...xv3xc3 input, or array of matrices assumed"""
    ai = np.empty_like(a)
    for i in range(3):
        ai[..., i, :] = np.cross(a[..., i - 2, :], a[..., i - 1, :])
    return ai


def inverse_transpose(a):
    """
    efficiently compute the inverse-transpose for stack of 3x3 matrices
    """
    ai = adjoint(a)
    det = dot(ai, a).mean(axis=-1)
    return ai / det[..., None, None]


def inverse(a):
    """inverse of a stack of 3x3 matrices"""
    return np.swapaxes(inverse_transpose(a), -1, -2)


def dot(a, b):
    """dot arrays of vecs; contract over last indices"""
    return np.einsum('...i,...i->...', a, b)


def adjoint_torch(a):
    ai = a.clone()
    for i in range(3):
        ai[..., i, :] = torch.cross(a[..., i - 2, :], a[..., i - 1, :])
    return ai


def inverse_transpose_torch(a):
    inv = adjoint_torch(a)
    det = dot_torch(inv, a).mean(dim=-1)
    return inv / det[:, None, None]


def inverse_torch(a):
    return inverse_transpose_torch(a).transpose(1, 2)


def dot_torch(a, b):
    a_view = a.view(-1, 1, 3)
    b_view = b.contiguous().view(-1, 3, 1)
    out = torch.bmm(a_view, b_view)
    out_view = out.view(a.size()[:-1])
    return out_view


if __name__ == "__main__":
    A = np.random.rand(2, 3, 3)
    AI = inverse(A)

    A_torch = torch.from_numpy(A)

    AI_torch = inverse_torch(A_torch)
    print(AI)
    print(AI_torch)
