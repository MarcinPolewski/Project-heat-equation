import argparse
import json
import pathlib

import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=pathlib.Path, required=True, help="Path to simulation config"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    dt=0.01
    dx=1
    plots=10
    T=1
    x_size=10
    y_size=10
    t=0
    default_val=0
    stop=0
    wait=1
    alpha=0.24
    points = [
        [5, 5, 10000]
    ]
    with open(args.config, "r") as f:
        config = json.load(f)
        if 'dt' in config:
            dt = config['dt']
        if 'plots' in config:
            plots = config['plots']
        if 'T' in config:
            T = config['T']
        if 'x_size' in config:
            x_size = config['x_size']
        if 'y_size' in config:
            y_size = config['y_size']
        if 'alpha' in config:
            alpha = config['alpha']
        if 'points' in config:
            points = config['points']
        if 'default_val' in config:
            default_val = config['default_val']
        if 'wait' in config:
            default_val = config['wait']

    u=np.full([x_size,y_size], default_val)

    for x,y,temprature in points:
        u[x][y]=temprature
    max_temp=np.max(u)
    min_temp=np.min(u)

    plt.show()

    while t<T:
        if t>=stop:
            p=plt.imshow(u,cmap='coolwarm',vmin=min_temp,vmax=max_temp)
            plt.colorbar(p)
            plt.pause(wait)
            plt.clf()
            stop+=T/plots

        u_new = u.copy()
        for j in range(1, y_size - 1):
            for i in range(1, x_size - 1):
                u_new[i, j] = u[i, j] + alpha * dt / dx ** 2 * (
                        u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1] - 4 * u[i, j]
                )
        u = u_new
        t+=dt