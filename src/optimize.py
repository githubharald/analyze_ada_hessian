from typing import Callable, List

import numpy as np
import torch

from ada_hessian import AdaHessian


def optimize(f: Callable,
             v0: List[float],
             num_iter: int,
             lr: float,
             bg: float,
             bh: float,
             k: float,
             num_samples: int) -> np.ndarray:
    """
    Optimize function f(v) with start value v0=(x0, y0) and num_iter iterations.
    num_samples specifies the number of H*v products used to approximate Hessian diagonal elements per step.
    Other parameters: learning rate (lr), beta for gradient (bg), beta for Hessian (bg), Hessian Power (k).
    See AdaHessian paper for details.
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
        print(f'#{i}: v={v.data}')
        path.append(v.clone().detach().numpy())

    return np.stack(path)
