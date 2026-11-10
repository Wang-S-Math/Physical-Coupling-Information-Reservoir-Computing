"""Parallel reservoir computing (PRC) baseline used in the PCI-RC paper.

The neighboring state component is selected according to the coupling form:
- Lorenz (x-x): x_j (component 0)
- Lorenz (x-y): y_j (component 1)
- Rössler (x-x): x_j (component 0)

Run this script from any working directory. Predictions are saved using the
filenames expected by figure.ipynb under result.data/PRC/.
"""

import shutil
from pathlib import Path

import numpy as np
from numpy import zeros, tanh

from rcnp import rcnp
from prc_configs import get_prc_config


PROJECT_DIR = Path(__file__).resolve().parent


def _project_path(relative_path):
    """Resolve a repository-relative path against the project directory."""
    path = Path(relative_path)
    return path if path.is_absolute() else PROJECT_DIR / path


def _is_git_lfs_pointer(path):
    """Return True if a file is a Git LFS pointer rather than its binary data."""
    try:
        with open(path, "rb") as f:
            header = f.read(200)
        return b"version https://git-lfs.github.com/spec/v1" in header
    except OSError:
        return False


def _load_npy(path, description):
    """Load a real .npy file and give a useful error for Git LFS pointers."""
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    if _is_git_lfs_pointer(path):
        raise RuntimeError(
            f"{description} is only a Git LFS pointer, not the actual .npy file:\n"
            f"  {path}\n"
            "Please obtain the Git LFS objects (for example, run `git lfs pull` "
            "inside a Git clone of the repository) and run this script again."
        )
    try:
        return np.load(path, allow_pickle=False)
    except Exception as exc:
        raise RuntimeError(f"Could not load {description}: {path}\n{exc}") from exc


def qiufei0(ww):
    """Return the neighbor indices for each node (all nonzero weights)."""
    ww = np.asarray(ww)
    if ww.ndim != 2 or ww.shape[0] != ww.shape[1]:
        raise ValueError(f"The weighted adjacency matrix must be square; got {ww.shape}.")
    if not np.all(np.isfinite(ww)):
        raise ValueError("The weighted adjacency matrix contains NaN or infinite values.")

    neighbors_by_node = []
    for i in range(ww.shape[0]):
        # Nonzero entries include both positive and negative inferred weights.
        neighbors_by_node.append(np.flatnonzero(ww[i] != 0))
    return neighbors_by_node


def CCB4_2_2_2(
    y1,
    y2,
    ww,
    nn=500,
    grla=1e-6,
    grg=0.7,
    pbj=0.95,
    xshl=0.03,
    grso=1.2,
    she=1000,
    ycn=10000,
    oo1=1,
    oo2=0.5,
    neighbor_component=0,
):
    """Train and run the PRC model.

    Parameters
    ----------
    y1 : ndarray, shape (n_train, 3 * n_nodes)
        Training trajectories, flattened node by node as (x_i, y_i, z_i).
    y2 : ndarray, shape (n_test, 3 * n_nodes)
        Reference test trajectories used only to calculate VPS.
    ww : ndarray, shape (n_nodes, n_nodes)
        Weighted adjacency matrix. Row i contains the weights used as inputs
        from the neighbors of target node i.
    neighbor_component : int
        Neighbor-state component supplied to each reservoir: 0 for x_j,
        1 for y_j, and 2 for z_j. This is selected from the system's coupling
        form in prc_configs.py.
    """
    y1 = np.asarray(y1, dtype=float)
    y2 = np.asarray(y2, dtype=float)
    ww = np.asarray(ww, dtype=float)

    if y1.ndim != 2 or y2.ndim != 2:
        raise ValueError(f"Training and test data must be 2-D; got {y1.shape}, {y2.shape}.")
    if y1.shape[1] % 3 != 0:
        raise ValueError(
            f"The flattened data width must be 3 * number_of_nodes; got {y1.shape[1]}."
        )
    if y2.shape[1] != y1.shape[1]:
        raise ValueError(
            f"Training/test data widths differ: {y1.shape[1]} vs {y2.shape[1]}."
        )
    if not np.all(np.isfinite(y1)) or not np.all(np.isfinite(y2)):
        raise ValueError("Training or test data contain NaN or infinite values.")
    if neighbor_component not in (0, 1, 2):
        raise ValueError("neighbor_component must be 0 (x), 1 (y), or 2 (z).")

    l1, l2 = y1.shape
    mn = l2 // 3
    if ww.shape != (mn, mn):
        raise ValueError(
            f"Weighted adjacency matrix shape {ww.shape} does not match "
            f"the data node count ({mn}, {mn})."
        )
    if not (0 <= she < l1 - 1):
        raise ValueError(f"she must satisfy 0 <= she < training length - 1; got {she}.")
    if ycn is None:
        ycn = y2.shape[0]
    ycn = int(ycn)
    if ycn <= 0 or ycn > y2.shape[0]:
        raise ValueError(f"Prediction length ycn={ycn} must be in [1, {y2.shape[0]}].")

    component_name = ("x", "y", "z")[neighbor_component]
    print(
        f"Starting PRC: Nodes={mn}, nn={nn}, training length={l1}, "
        f"prediction length={ycn}, neighbor component={component_name}_j"
    )

    B = []
    max_val = np.max(np.abs(y1))
    if max_val == 0:
        max_val = 1.0
    grgs = grso / max_val

    # The row-wise convention is ww[i, j]: input from node j to node i.
    wl = qiufei0(ww)

    # 1. Training stage
    for ii in range(mn):
        B1 = rcnp()
        u_self = y1[:, ii * 3 : (ii + 1) * 3]

        neighbor_inputs = []
        for jj in wl[ii]:
            w_val = ww[ii, jj]
            # Use the state component specified by the system's coupling form.
            u_neigh = y1[:, jj * 3 + neighbor_component] * w_val
            neighbor_inputs.append(u_neigh.reshape(-1, 1))

        if neighbor_inputs:
            u_full = np.hstack([u_self] + neighbor_inputs)
        else:
            u_full = u_self

        input_dim = u_full.shape[1]
        try:
            B1.win = B1.fwin3(nn, input_dim, 3, 1, grgs)
        except AttributeError:
            # Retain compatibility with rcnp implementations without fwin3.
            B1.win = B1.fwin(nn, input_dim, grgs)

        current_xshl = xshl
        if nn * current_xshl < 1.5:
            current_xshl = 3.0 / nn
        B1.setwres(nn, current_xshl, pbj)
        B1.bk = B1.fbk(nn, oo1, oo2)

        B1.train(u_full, grg)
        B.append(B1)

    # Fit each reservoir to predict the next state of its own node.
    for ii in range(mn):
        target = y1[she:, ii * 3 : (ii + 1) * 3]
        states = B[ii].RR[she - 1 : -1]
        B[ii].getwout(target, states, grla)

    print("Readout weights fitted.")

    # 2. Autonomous prediction stage
    yc = zeros((ycn, l2))
    rrduo = zeros((mn, nn))
    for ii in range(mn):
        rrduo[ii] = B[ii].rr.copy()

    for zz in range(ycn):
        # Predict all nodes first, then update all reservoirs synchronously.
        for ii in range(mn):
            yc[zz, ii * 3 : (ii + 1) * 3] = B[ii].out @ rrduo[ii]

        for ii in range(mn):
            u_self_t = yc[zz, ii * 3 : (ii + 1) * 3]
            u_neigh_list = []
            for jj in wl[ii]:
                w_val = ww[ii, jj]
                val_neigh = yc[zz, jj * 3 + neighbor_component] * w_val
                u_neigh_list.append(val_neigh)

            if u_neigh_list:
                u_full_t = np.hstack([u_self_t, np.asarray(u_neigh_list)])
            else:
                u_full_t = u_self_t

            linear_part = B[ii].win @ u_full_t + B[ii].wres @ rrduo[ii] + B[ii].bk
            rrduo[ii] = (1 - B[ii].grg) * rrduo[ii] + B[ii].grg * tanh(linear_part)

    print("Prediction completed.")
    vps = B[0].fbijiao(yc, y2[:ycn], 0.5)
    print(f"VPS: {vps} steps")
    return B, yc


def save_prediction(predictions, output_file, backup_existing=True):
    """Save predictions in the .npy format and filenames used by figure.ipynb.

    If an archived result already exists, preserve its first copy as a .bak file
    before replacing it with the newly generated prediction.
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if output_file.exists() and backup_existing:
        backup_file = output_file.with_name(output_file.name + ".before_prc_correction.bak")
        if not backup_file.exists():
            shutil.copy2(output_file, backup_file)
            print(f"Previous result backed up to: {backup_file}")

    np.save(output_file, np.asarray(predictions))
    print(f"Prediction saved to: {output_file}")


def main():
    # Select the system and network size here.
    # Options: "llz_xx", "llz_xy", "lsl".
    SYSTEM = "llz_xy"
    M = 10

    cfg = get_prc_config(SYSTEM, M)
    data_file = _project_path(cfg["data_file"])
    weight_file = _project_path(cfg["weight_file"])
    output_file = _project_path(cfg["prediction_file"])

    raw_data = _load_npy(data_file, "Input data")
    weight_matrix = _load_npy(weight_file, "Reconstructed weighted adjacency matrix")

    print("=" * 72)
    print(f"System: {SYSTEM}")
    print(f"Network size: M={M}")
    print(f"Data file: {data_file}")
    print(f"Weighted network file: {weight_file}")
    component_name = ("x", "y", "z")[cfg["neighbor_component"]]
    print(f"Neighbor input component: {component_name}_j")
    print(f"Output file: {output_file}")
    print("=" * 72)

    if raw_data.ndim == 3:
        steps, n_nodes, n_dim = raw_data.shape
        if n_dim != 3:
            raise ValueError(f"Expected 3 state variables per node, got {n_dim}.")
        if n_nodes != M:
            raise ValueError(f"Data contains {n_nodes} nodes, but M={M} was selected.")
        flat_data = raw_data.reshape(steps, n_nodes * n_dim)
    elif raw_data.ndim == 2:
        flat_data = raw_data
        if flat_data.shape[1] != 3 * M:
            raise ValueError(
                f"Expected {3 * M} columns for M={M}, got {flat_data.shape[1]}."
            )
    else:
        raise ValueError(f"Unexpected data shape: {raw_data.shape}")

    train_len = int(cfg["train_len"])
    if len(flat_data) <= train_len:
        raise ValueError(
            f"Insufficient data length: {len(flat_data)} <= {train_len}; "
            "the configured training segment is not available."
        )

    y1_train = flat_data[:train_len]
    available_test = flat_data[train_len:]
    pred_len = min(int(cfg["pred_len"]), len(available_test))
    y2_test = available_test[:pred_len]

    print(f"Training length: {len(y1_train)}")
    print(f"Prediction length: {len(y2_test)}")

    _, predictions = CCB4_2_2_2(
        y1_train,
        y2_test,
        weight_matrix,
        nn=cfg["nn"],
        grla=cfg["grla"],
        grg=cfg["grg"],
        pbj=cfg["pbj"],
        xshl=cfg["xshl"],
        grso=cfg["grso"],
        she=cfg["she"],
        ycn=len(y2_test),
        neighbor_component=cfg["neighbor_component"],
    )

    if predictions.shape != y2_test.shape:
        raise RuntimeError(
            f"Prediction shape {predictions.shape} does not match test shape {y2_test.shape}."
        )

    # Save under the same filenames and folder read by figure.ipynb.
    save_prediction(predictions, output_file, backup_existing=True)
    print(f"Prediction shape: {predictions.shape}")
    print("Done.")


if __name__ == "__main__":
    main()