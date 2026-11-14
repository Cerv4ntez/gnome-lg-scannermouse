"""Solving segment positions from pairwise measurements (used by mapper.py).

Each measurement says where one segment lies relative to another. `solve`
finds the positions that agree best with all of them, down-weighting
measurements that disagree strongly with the rest.

Poses are (x, y, theta): frame centre in full-resolution pixels, rotation in rad.
"""

import math

import numpy as np
from scipy import optimize, sparse

ANGLE_SCALE = 150.0           # px per radian when mixing angle and position residuals
REWEIGHT_ROUNDS = 3
OUTLIER_PX = 8.0              # measurements disagreeing by more than this get down-weighted


def _rot(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def relative(pi, pj):
    """Pose of j expressed in the frame of i."""
    t = _rot(-pi[2]) @ (pj[:2] - pi[:2])
    return np.array([t[0], t[1], pj[2] - pi[2]])


def solve(poses, edges):
    """Least-squares poses from (i, j, measured relative pose, weight) edges; node 0 is fixed."""
    poses = np.asarray(poses, float)
    n = len(poses)
    if n < 2 or not edges:
        return poses
    I = np.array([e[0] for e in edges])
    J = np.array([e[1] for e in edges])
    meas = np.array([e[2] for e in edges], float)
    wgt = np.array([e[3] for e in edges], float)[:, None]

    def residuals(flat):
        p = np.vstack([poses[0], flat.reshape(-1, 3)])
        d = p[J, :2] - p[I, :2]
        c, s = np.cos(p[I, 2]), np.sin(p[I, 2])
        r = np.empty((len(edges), 3))
        r[:, 0] = c * d[:, 0] + s * d[:, 1] - meas[:, 0]
        r[:, 1] = -s * d[:, 0] + c * d[:, 1] - meas[:, 1]
        r[:, 2] = ((p[J, 2] - p[I, 2] - meas[:, 2] + np.pi) % (2 * np.pi) - np.pi) * ANGLE_SCALE
        return (wgt * r).ravel()

    rows = np.arange(len(edges)) * 3

    def jacobian(flat):
        p = np.vstack([poses[0], flat.reshape(-1, 3)])
        d = p[J, :2] - p[I, :2]
        c, s = np.cos(p[I, 2]), np.sin(p[I, 2])
        w = wgt[:, 0]
        # (row offset, node, variable, value) for every non-zero derivative
        entries = [
            (0, J, 0, c), (0, J, 1, s), (0, I, 0, -c), (0, I, 1, -s), (0, I, 2, -s * d[:, 0] + c * d[:, 1]),
            (1, J, 0, -s), (1, J, 1, c), (1, I, 0, s), (1, I, 1, -c), (1, I, 2, -c * d[:, 0] - s * d[:, 1]),
            (2, J, 2, np.full(len(edges), ANGLE_SCALE)), (2, I, 2, np.full(len(edges), -ANGLE_SCALE)),
        ]
        r, cidx, v = [], [], []
        for off, node, var, val in entries:
            keep = node > 0                    # node 0 is held fixed
            r.append(rows[keep] + off)
            cidx.append(3 * (node[keep] - 1) + var)
            v.append((w * val)[keep])
        return sparse.csr_matrix((np.concatenate(v), (np.concatenate(r), np.concatenate(cidx))),
                                 shape=(3 * len(edges), 3 * (n - 1)))

    # Plain least squares, re-solved with strongly disagreeing measurements
    # down-weighted, which limits the damage of a wrong match. (scipy's robust
    # losses stall on this problem.)
    base = wgt.copy()
    x = poses[1:].ravel()
    for _ in range(REWEIGHT_ROUNDS):
        x = optimize.least_squares(residuals, x, jac=jacobian, x_scale='jac',
                                   method='trf', max_nfev=100).x
        err = np.linalg.norm((residuals(x) / np.repeat(wgt[:, 0], 3)).reshape(-1, 3), axis=1)
        wgt[:, 0] = base[:, 0] * np.minimum(1.0, OUTLIER_PX / np.maximum(err, 1e-9))
    out = np.vstack([poses[0], x.reshape(-1, 3)])
    out[:, 2] = (out[:, 2] + np.pi) % (2 * np.pi) - np.pi
    return out


def solve_lines(poses, edges, pieces, n_lines):
    """As `solve`, plus straight lines: `pieces` are (segment, ends, line) with the two
    end points of a line segment in that segment's frame (centred coordinates, (2, 2)),
    and `line` the index of the straight line on the page it belongs to. Every piece
    is pulled onto its line; the lines' positions are solved too. Node 0 is fixed."""
    poses = np.asarray(poses, float)
    n = len(poses)
    I = np.array([e[0] for e in edges])
    J = np.array([e[1] for e in edges])
    meas = np.array([e[2] for e in edges], float)
    wgt = np.array([e[3] for e in edges], float)[:, None]
    seg = np.array([p[0] for p in pieces])
    ends = np.array([p[1] for p in pieces], float)
    line = np.array([p[2] for p in pieces])

    def world(p):
        c, s = np.cos(p[seg, 2])[:, None], np.sin(p[seg, 2])[:, None]
        return (c * ends[:, :, 0] - s * ends[:, :, 1] + p[seg, 0][:, None],
                s * ends[:, :, 0] + c * ends[:, :, 1] + p[seg, 1][:, None])

    # starting lines: direction and offset fitted to their pieces
    wx, wy = world(poses)
    start = []
    for k in range(n_lines):
        pts = np.c_[wx[line == k].ravel(), wy[line == k].ravel()]
        centre = pts.mean(0)
        u = np.linalg.svd(pts - centre)[2][0]
        phi = math.atan2(u[1], u[0])
        start.append((phi, -math.sin(phi) * centre[0] + math.cos(phi) * centre[1]))

    def unpack(x):
        return np.vstack([poses[0], x[:3 * (n - 1)].reshape(-1, 3)]), x[3 * (n - 1):].reshape(-1, 2)

    def residuals(x):
        p, L = unpack(x)
        d = p[J, :2] - p[I, :2]
        c, s = np.cos(p[I, 2]), np.sin(p[I, 2])
        r = np.empty((len(edges), 3))
        r[:, 0] = c * d[:, 0] + s * d[:, 1] - meas[:, 0]
        r[:, 1] = -s * d[:, 0] + c * d[:, 1] - meas[:, 1]
        r[:, 2] = ((p[J, 2] - p[I, 2] - meas[:, 2] + np.pi) % (2 * np.pi) - np.pi) * ANGLE_SCALE
        wx, wy = world(p)
        phi, rho = L[line, 0][:, None], L[line, 1][:, None]
        off = -np.sin(phi) * wx + np.cos(phi) * wy - rho    # distance of each end point from its line
        return np.r_[(wgt * r).ravel(), off.ravel()]

    rows_e, rows_l = 3 * len(edges), 2 * len(pieces)
    sparsity = sparse.lil_matrix((rows_e + rows_l, 3 * (n - 1) + 2 * n_lines), dtype=np.uint8)
    for e, (i, j) in enumerate(zip(I, J)):
        for node in (i, j):
            if node > 0:
                sparsity[3 * e:3 * e + 3, 3 * (node - 1):3 * node] = 1
    for q, (k, l) in enumerate(zip(seg, line)):
        if k > 0:
            sparsity[rows_e + 2 * q:rows_e + 2 * q + 2, 3 * (k - 1):3 * k] = 1
        sparsity[rows_e + 2 * q:rows_e + 2 * q + 2, 3 * (n - 1) + 2 * l:3 * (n - 1) + 2 * l + 2] = 1
    x = np.r_[poses[1:].ravel(), np.array(start).ravel()]
    x = optimize.least_squares(residuals, x, jac_sparsity=sparsity, loss='soft_l1', f_scale=OUTLIER_PX / 2,
                               x_scale='jac', method='trf', max_nfev=200).x
    out = unpack(x)[0]
    out[:, 2] = (out[:, 2] + np.pi) % (2 * np.pi) - np.pi
    return out
