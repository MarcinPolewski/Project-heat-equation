import argparse
import json
import pathlib

import numpy as np
import matplotlib.pyplot as plt


class HeatEquationSolver:

    def next_step_FTCS(self, u, alpha, dt, dx):
        u_new = u.copy()
        rows, cols = u.shape
        for j in range(1, cols - 1):
            for i in range(1, rows - 1):
                u_new[i, j] = u[i, j] + alpha * dt / dx**2 * (
                    u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1] - 4 * u[i, j]
                )
        return u_new

    def plot_heat_map_2d(self, u, min_temp, max_temp):
        p = plt.imshow(u, cmap="coolwarm", vmin=min_temp, vmax=max_temp)
        plt.colorbar(p)
        plt.pause(1)
        plt.clf()

    def solve(self, u, alpha, dt, dx, T, plots):

        plt.show()

        max_temp = np.max(u)
        min_temp = np.min(u)

        stop = 0
        t = 0
        while t < T:
            if t >= stop:
                self.plot_heat_map_2d(u, min_temp, max_temp)
                stop += T / plots
            u = self.next_step_FTCS(u, alpha, dt, dx)
            t += dt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=pathlib.Path, required=True, help="Path to simulation config"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dt = 0.01
    dx = 1
    plots = 10
    T = 1
    x_size = 10
    y_size = 10
    default_val = 0
    alpha = 0.24
    points = [[5, 5, 10000]]
    with open(args.config, "r") as f:
        config = json.load(f)
        if "dt" in config:
            dt = config["dt"]
        if "plots" in config:
            plots = config["plots"]
        if "T" in config:
            T = config["T"]
        if "x_size" in config:
            x_size = config["x_size"]
        if "y_size" in config:
            y_size = config["y_size"]
        if "alpha" in config:
            alpha = config["alpha"]
        if "points" in config:
            points = config["points"]
        if "default_val" in config:
            default_val = config["default_val"]
        if "wait" in config:
            default_val = config["wait"]

    u = np.full([x_size, y_size], default_val)

    for x, y, temprature in points:
        u[x][y] = temprature

    solver = HeatEquationSolver()
    solver.solve(u, alpha, dt, dx, T, plots)


if "__main__" == __name__:
    main()
