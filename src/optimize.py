import torch
from ada_hessian import AdaHessian
import numpy as np


def optimize(f, v0, num_iter, lr, bg, bh, k, num_samples):
    """
    optimize function f(v) with start value v0=(x0, y0) and num_iter iterations
    num_samples specifies the number of H*v products used to approx. Hessian diagonal elements per step
    other parameter: learning rate (lr), beta for gradient (bg), beta for Hessian (bg), Hessian Power (k)
    """
    torch.manual_seed(0)

    v = torch.tensor(v0, dtype=torch.float64, requires_grad=True)
    optimizer = AdaHessian([v], lr=lr, betas=(bg, bh), n_samples=num_samples, hessian_power=k)

    path = [v0]
    for i in range(num_iter):
        # compute function value, compute gradient of y(v) w.r.t. v, apply single optimizer step
        y = f(v)
        y.backward(create_graph=True)
        optimizer.step()
        optimizer.zero_grad()

        # add current value to path
        print(f'#{i}: v={v}')
        path.append(v.clone().detach().numpy())

    return np.stack(path)
