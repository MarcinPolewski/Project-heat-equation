import argparse
import json
import pathlib

import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


class HeatEquationSolver:

    def __init__(self, wait, numeric_method=None, plot_method=None):
        if numeric_method is None:
            numeric_method = self.next_step_FTCS
        if plot_method is None:
            plot_method = self.plot_heat_map_2d
        self.numeric_method = numeric_method
        self.plot_method = plot_method
        self.wait_time = wait

    @staticmethod
    def next_step_FTCS(u, alpha, dt, dx):
        u_new = u.copy()
        rows, cols = u.shape
        for j in range(1, cols - 1):
            for i in range(1, rows - 1):
                u_new[i, j] = u[i, j] + alpha * dt / dx**2 * (
                    u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1] - 4 * u[i, j]
                )
        return u_new

    @staticmethod
    def next_step_BTCS(u, alpha, dt, dx):
        rows, cols = u.shape
        size = rows * cols
        coeff = alpha * dt / dx**2

        # Create the coefficient matrix
        # laplace_matrix represents discretized laplace operator
        # for heat equation in 2D. It encodes the relationships between each grid point and its neighbors
        main_diag = (1 + 4 * coeff) * np.ones(size)
        off_diag = -coeff * np.ones(size - 1)
        laplace_matrix = diags(
            [
                main_diag,
                off_diag,
                off_diag,
                -coeff * np.ones(size - rows),
                -coeff * np.ones(size - rows),
            ],
            [0, 1, -1, rows, -rows],
            format="csr",
        )

        # solve equation system
        u_flat = u.flatten()
        u_new_flat = spsolve(laplace_matrix, u_flat)
        return u_new_flat.reshape(rows, cols)

    @staticmethod
    def plot_heat_map_2d(u, min_temp, max_temp, wait_time):
        if not hasattr(HeatEquationSolver, "_fig2d"):
            HeatEquationSolver._fig2d = plt.figure()
        plt.figure(HeatEquationSolver._fig2d.number)
        plt.clf()
        p = plt.imshow(u, cmap="coolwarm", vmin=min_temp, vmax=max_temp)
        plt.colorbar(p)
        plt.draw()
        plt.pause(wait_time)

    @staticmethod
    def plot_heat_map_3d(u, min_temp, max_temp, wait_time):
        if not hasattr(HeatEquationSolver, "_fig3d"):
            HeatEquationSolver._fig3d = plt.figure()
        fig = HeatEquationSolver._fig3d
        fig.clf()
        ax = fig.add_subplot(111, projection="3d")
        rows, cols = u.shape
        x = np.arange(rows)
        y = np.arange(cols)
        x, y = np.meshgrid(x, y)
        ax.plot_surface(x, y, u, cmap="coolwarm", vmin=min_temp, vmax=max_temp)
        ax.set_zlim(min_temp, max_temp)

        plt.draw()
        plt.pause(wait_time)

    def solve(self, u, alpha, dt, dx, T, plots):

        plt.show()

        max_temp = np.max(u)
        min_temp = np.min(u)

        stop = 0
        t = 0
        while t < T:
            if t >= stop:
                self.plot_method(u, min_temp, max_temp, self.wait_time)
                stop += T / plots
            u = self.numeric_method(u, alpha, dt, dx)
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
    alpha = 0.24
    points = [[5, 5, 10000]]
    default_val = 0.001
    wait = 1
    numeric_method = HeatEquationSolver.next_step_FTCS
    plot_method = HeatEquationSolver.plot_heat_map_2d
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
            wait = config["wait"]
        if "numeric_method" in config:
            if config["numeric_method"] == "FTCS":
                numeric_method = HeatEquationSolver.next_step_FTCS
            elif config["numeric_method"] == "BTCS":
                numeric_method = HeatEquationSolver.next_step_BTCS
            else:
                raise ValueError("Invalid numeric method")
        if "plot_method" in config:
            if config["plot_method"] == "2d":
                plot_method = HeatEquationSolver.plot_heat_map_2d
            elif config["plot_method"] == "3d":
                plot_method = HeatEquationSolver.plot_heat_map_3d
            else:
                raise ValueError("Invalid plot method")

    u = np.full([x_size, y_size], default_val)

    for x, y, temprature in points:
        u[x][y] = temprature

    solver = HeatEquationSolver(wait, numeric_method, plot_method)
    solver.solve(u, alpha, dt, dx, T, plots)


if "__main__" == __name__:
    main()
