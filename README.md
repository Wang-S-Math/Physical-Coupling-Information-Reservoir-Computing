# PCI-RC Code and Data

This repository contains the code and numerical data used in the manuscript **“Weighted Network Reconstruction via Physical Coupling Information Reservoir Computing.”** The saved `.npy` files can be used directly to reproduce the figures in the paper, while the Python scripts can be used to rerun the corresponding reconstruction and prediction experiments. For convenience, the main experiment scripts do not overwrite the archived paper results.

The six PCI-RC reconstruction scripts correspond to the three dynamical systems and two topology settings used in the paper:

```text
llz.(x-x).kn.py   Lorenz x-x coupling, known topology
llz.(x-x).un.py   Lorenz x-x coupling, unknown topology
llz.xy.kn.py      Lorenz x*y coupling, known topology
llz.xy.un.py      Lorenz x*y coupling, unknown topology
lsl.kn.py         Rossler x-x coupling, known topology
lsl.un.py         Rossler x-x coupling, unknown topology
```

`configs.py` stores the experiment settings shared by these six scripts, including the corresponding data paths and size-dependent parameters.

## Repository folders

```text
generate.data/          Input datasets used by the PCI-RC experiments, including the
                        time-series data and ground-truth adjacency matrices for
                        different network sizes.

mathod/                 Auxiliary system-specific scripts for the three dynamical
                        systems used in the experimental workflow.

noise_results_llz_xy/   Saved outputs of the observational-noise robustness experiment
                        for the nonlinear product-coupled Lorenz system, including the
                        numerical results and the corresponding figure.

result.data/            Archived reconstruction and dynamical-prediction results used
                        by figure.ipynb to reproduce the manuscript figures.
```

To rerun one reconstruction experiment, run the corresponding Python script from the repository root. For example:

```bash
python llz.xy.un.py
```

If you want to reproduce a different network size, only change the value of `M` near the bottom of the selected script:

```python
# ============================================================
# Network size / 网络规模
# If you want reconstruction results for a different network size,
# change the value of M below.  (Change me)
# 如果想查看不同网络规模的重构结果，请修改下面的 M 值。（修改这里）
# ============================================================
M = 10  # Change me / 修改这里
```

`PRC.py` contains the PRC baseline used for comparison with PCI-RC. The system and network size can be selected by changing:

```python
SYSTEM = "llz_xx"  # "llz_xx", "llz_xy", or "lsl"
M = 10
```

The shared PRC settings are stored in `prc_configs.py`, and `rcnp.py` contains the reservoir-computing functions used by the PRC baseline.

`AUC.py` is used for the threshold-sensitivity experiment. Running it generates the threshold-sensitivity numerical results and figure:

```text
threshold_ROC.csv
threshold_ROC.pdf
```

`noise.py` is used for the observational-noise robustness experiment. Its outputs are stored in:

```text
noise_results_llz_xy/
```

`figure.ipynb` is used to reproduce the manuscript figures from the archived numerical results. please open the notebook and run the cells from top to bottom.
Install the required Python packages with:

```bash
pip install -r requirements.txt
```