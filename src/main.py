import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse

from ada_hessian import AdaHessian
from plot import plot_path, plot_function, show_plot
from optimize import optimize


def make_func(s):
    "parse the passed function string"
    def func(v):
        return eval(s)
    return func


def parse_args():
    "parse command line arguments"
    ap = argparse.ArgumentParser()
    ap.add_argument('--func', required=False, type=str, default='10*v[0]**2 + v[1]**4')
    ap.add_argument('--num_iter', required=False, type=int, default=10)
    ap.add_argument('--start', required=False, nargs='+', type=float, default=(1, 1))
    ap.add_argument('--window', required=False, nargs='+', type=float, default=(-2, 2, -2, 2))
    ap.add_argument('--lr', required=False, type=float, default=0.1)
    ap.add_argument('--beta_g', required=False, type=float, default=0.5)
    ap.add_argument('--beta_h', required=False, type=float, default=0.5)
    ap.add_argument('--hessian_pow', required=False, type=float, default=1)
    ap.add_argument('--num_samples', required=False, type=int, default=1)


    parsed = ap.parse_args()

    assert len(parsed.start) == 2
    assert len(parsed.window) == 4
    assert parsed.window[0] < parsed.window[1] and parsed.window[2] < parsed.window[3]

    return parsed


def main():

    # parse command line arguments and parse function
    parsed = parse_args()
    func = make_func(parsed.func)

    # optimize 2d parameter vector v of function
    path = optimize(func, parsed.start, parsed.num_iter, parsed.lr, parsed.beta_g, parsed.beta_h, parsed.hessian_pow, parsed.num_samples)

    # plot function and steps of optimizer
    plot_function(func, parsed.window)
    plot_path(path)
    show_plot()



if __name__ == '__main__':
    main()
