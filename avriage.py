
# -*- coding: utf-8 -*-
"""
PCI-RC 20次实验结果汇总

输出：
1. all_20runs_metrics_mean_std.csv
   所有系统、拓扑、振子规模的指标均值和标准差

2. Table_II_unknown_topology.csv
   未知拓扑的 Error、TPR、TNR、Precision
   用于更新 Table II

3. VPS_M10_KN_for_Fig8.csv
   三个系统 M=10、已知拓扑的 VPS
   用于更新 Fig.8

4. VPS_M10_all_topologies.csv
   M=10 的 KN/UN VPS 分开汇总
"""

import csv
import re
import statistics
from pathlib import Path


# =====================================================
# 1. 实验配置
# =====================================================

EXPECTED_RUNS = 20

SYSTEM_INFO = {
    "llz_xx": {
        "label": "Lorenz (x-x)",
        "folder": "llz.(x-x)"
    },
    "llz_xy": {
        "label": "Lorenz (x-y)",
        "folder": "llz.xy"
    },
    "lsl": {
        "label": "Rossler (x-x)",
        "folder": "lsl"
    }
}

# 预期运行的网络规模
EXPECTED_SIZES = {
    ("llz_xx", "known"): [5, 10, 20, 50, 100],
    ("llz_xx", "unknown"): [5, 10, 20, 50, 100],

    ("llz_xy", "known"): [5, 10, 20, 50, 100],
    ("llz_xy", "unknown"): [5, 10, 20, 50, 100, 700],

    ("lsl", "known"): [5, 10, 20, 50, 100],
    ("lsl", "unknown"): [5, 10, 20, 50, 100]
}

# CSV文件名识别
FILE_RE = re.compile(
    r"^(llz_xx|llz_xy|lsl)_(known|unknown)_M(\d+)_20runs_raw\.csv$",
    re.I
)

# 指标名称兼容
ALIASES = {
    "run": {"run"},
    "Error": {"error", "weight_error"},
    "TPR": {"tpr"},
    "TNR": {"tnr"},
    "Precision": {"precision", "precise", "preecise"},
    "VPS": {"vps"}
}


# =====================================================
# 2. 自动定位 result.date
# =====================================================

def find_result_root():

    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd()

    candidates = [
        script_dir / "result.date",
        cwd / "result.date",
        script_dir.parent / "result.date",
        cwd.parent / "result.date"
    ]

    for path in dict.fromkeys(p.resolve() for p in candidates):

        if path.is_dir():
            return path

    raise FileNotFoundError(
        "找不到 result.date 文件夹。\n"
        "请将本脚本放在 E:\\PI-IRC 项目根目录，"
        "或从项目根目录运行。"
    )


# =====================================================
# 3. 查找CSV字段
# =====================================================

def find_col(fieldnames, metric):

    for name in fieldnames:

        if str(name).strip().lower() in ALIASES[metric]:
            return name

    return None


# =====================================================
# 4. 计算均值和样本标准差
# =====================================================

def stats(values):

    if len(values) != EXPECTED_RUNS:
        raise ValueError(
            f"应有20个数值，实际为{len(values)}个。"
        )

    mean = statistics.mean(values)
    std = statistics.stdev(values)

    return mean, std


# =====================================================
# 5. 读取单个20次实验CSV
# =====================================================

def read_raw_csv(path, topology, M):

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        reader = csv.DictReader(f)

        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if len(rows) != EXPECTED_RUNS:

        raise ValueError(
            f"{path}\n"
            f"应有20行，实际读取到{len(rows)}行。"
        )

    run_col = find_col(fieldnames, "run")

    if run_col is None:
        raise ValueError(f"{path}\n缺少run列。")

    try:

        rows.sort(
            key=lambda r: int(r[run_col])
        )

        run_ids = [
            int(r[run_col]) for r in rows
        ]

    except (TypeError, ValueError, KeyError) as exc:

        raise ValueError(
            f"{path}\nrun编号无法识别。"
        ) from exc

    if run_ids != list(range(1, 21)):

        raise ValueError(
            f"{path}\n"
            "run编号必须完整且唯一，为1至20。"
        )

    # 所有实验均统计Error
    metrics = ["Error"]

    # 未知拓扑额外统计三个网络指标
    if topology == "unknown":

        metrics += [
            "TPR",
            "TNR",
            "Precision"
        ]

    # 只有M=10计算VPS
    if M == 10:

        metrics += ["VPS"]

    result = {}

    for metric in metrics:

        col = find_col(fieldnames, metric)

        if col is None:

            raise ValueError(
                f"{path}\n缺少指标列：{metric}"
            )

        values = []

        for row in rows:

            raw = row.get(col, "")

            if raw is None or str(raw).strip() == "":

                raise ValueError(
                    f"{path}\n"
                    f"第{row[run_col]}次实验的{metric}为空。"
                )

            try:

                values.append(float(raw))

            except (TypeError, ValueError) as exc:

                raise ValueError(
                    f"{path}\n"
                    f"第{row[run_col]}次实验的"
                    f"{metric}不是数字：{raw}"
                ) from exc

        result[metric] = values

    return result


# =====================================================
# 6. 保存CSV
# =====================================================

def write_csv(path, columns, rows):

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=columns
        )

        writer.writeheader()
        writer.writerows(rows)


# =====================================================
# 7. 主程序
# =====================================================

def main():

    result_root = find_result_root()

    output_dir = (
        result_root / "20run_summary"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("读取目录：", result_root)
    print("汇总目录：", output_dir)

    raw_files = sorted(
        result_root.rglob("*_20runs_raw.csv")
    )

    if not raw_files:

        raise FileNotFoundError(
            f"{result_root}下没有找到20次实验原始CSV。"
        )

    all_rows = []
    table_rows = []
    vps_rows = []

    found = set()
    file_count = 0

    # =================================================
    # 遍历全部实验文件
    # =================================================

    for path in raw_files:

        if path.parent.name != "20run_metrics":
            continue

        match = FILE_RE.match(path.name)

        if not match:

            print("[跳过文件名不匹配]", path)
            continue

        system, topology, m_text = match.groups()

        system = system.lower()
        topology = topology.lower()
        M = int(m_text)

        key = (system, topology, M)

        if key in found:

            raise ValueError(
                f"发现重复文件：{key}"
            )

        found.add(key)

        expected_folder = SYSTEM_INFO[system]["folder"]

        if path.parent.parent.name != expected_folder:

            print(
                "[警告] 系统目录不一致：",
                path
            )

        values_by_metric = read_raw_csv(
            path,
            topology,
            M
        )

        file_count += 1

        label = SYSTEM_INFO[system]["label"]

        print(
            f"[读取成功] {label}, "
            f"{topology}, M={M}: "
            f"{', '.join(values_by_metric)}"
        )

        # ---------------------------------------------
        # 7.1 所有指标的长表
        # ---------------------------------------------

        for metric, values in values_by_metric.items():

            mean, std = stats(values)

            all_rows.append({

                "System": label,
                "System_Code": system,
                "Topology": topology,
                "M": M,
                "Metric": metric,
                "N": len(values),
                "Mean": mean,
                "Std_Sample": std,
                "Mean ± Std": (
                    f"{mean:.8g} ± {std:.8g}"
                ),
                "Min": min(values),
                "Max": max(values),
                "Source_File": path.name
            })

        # ---------------------------------------------
        # 7.2 Table II：未知拓扑
        # ---------------------------------------------

        if topology == "unknown":

            row = {
                "System": label,
                "System_Code": system,
                "M": M,
                "N": EXPECTED_RUNS
            }

            for metric in [
                "Error",
                "TPR",
                "TNR",
                "Precision"
            ]:

                mean, std = stats(
                    values_by_metric[metric]
                )

                row[f"{metric}_Mean"] = mean
                row[f"{metric}_Std"] = std

                row[f"{metric}_Mean±Std"] = (
                    f"{mean:.8g} ± {std:.8g}"
                )

            table_rows.append(row)

        # ---------------------------------------------
        # 7.3 M=10的VPS
        # ---------------------------------------------

        if M == 10 and "VPS" in values_by_metric:

            mean, std = stats(
                values_by_metric["VPS"]
            )

            vps_rows.append({

                "System": label,
                "System_Code": system,
                "Topology": topology,
                "M": M,
                "N": EXPECTED_RUNS,
                "VPS_Mean": mean,
                "VPS_Std": std,
                "VPS_Mean±Std": (
                    f"{mean:.8g} ± {std:.8g}"
                ),
                "Source_File": path.name
            })

    # =================================================
    # 8. 检查缺失文件
    # =================================================

    missing = []

    for (system, topology), sizes in EXPECTED_SIZES.items():

        folder = (
            result_root
            / SYSTEM_INFO[system]["folder"]
            / "20run_metrics"
        )

        for M in sizes:

            if (system, topology, M) not in found:

                missing.append(
                    folder /
                    f"{system}_{topology}_M{M}_20runs_raw.csv"
                )

    # =================================================
    # 9. 保存全部指标
    # =================================================

    all_rows.sort(
        key=lambda r: (
            r["System_Code"],
            r["Topology"],
            r["M"],
            r["Metric"]
        )
    )

    write_csv(

        output_dir /
        "all_20runs_metrics_mean_std.csv",

        [
            "System",
            "System_Code",
            "Topology",
            "M",
            "Metric",
            "N",
            "Mean",
            "Std_Sample",
            "Mean ± Std",
            "Min",
            "Max",
            "Source_File"
        ],

        all_rows
    )

    # =================================================
    # 10. 保存Table II数据
    # =================================================

    table_rows.sort(
        key=lambda r: (
            r["M"],
            r["System_Code"]
        )
    )

    write_csv(

        output_dir /
        "Table_II_unknown_topology.csv",

        [
            "System",
            "System_Code",
            "M",
            "N",

            "Error_Mean",
            "Error_Std",
            "Error_Mean±Std",

            "TPR_Mean",
            "TPR_Std",
            "TPR_Mean±Std",

            "TNR_Mean",
            "TNR_Std",
            "TNR_Mean±Std",

            "Precision_Mean",
            "Precision_Std",
            "Precision_Mean±Std"
        ],

        table_rows
    )

    # =================================================
    # 11. 保存M=10三个系统的VPS
    # =================================================

    vps_rows.sort(
        key=lambda r: (
            r["System_Code"],
            r["Topology"]
        )
    )

    vps_columns = [
        "System",
        "System_Code",
        "Topology",
        "M",
        "N",
        "VPS_Mean",
        "VPS_Std",
        "VPS_Mean±Std",
        "Source_File"
    ]

    write_csv(

        output_dir /
        "VPS_M10_all_topologies.csv",

        vps_columns,
        vps_rows
    )

    # Fig.8使用KN的VPS
    vps_kn_rows = [
        r for r in vps_rows
        if r["Topology"] == "known"
    ]

    vps_kn_rows.sort(
        key=lambda r: r["System_Code"]
    )

    write_csv(

        output_dir /
        "VPS_M10_KN_for_Fig8.csv",

        vps_columns,
        vps_kn_rows
    )

    # =================================================
    # 12. 控制台打印Table II数据
    # =================================================

    print("\n" + "=" * 105)

    print(
        "未知拓扑：20次实验均值 ± 样本标准差"
    )

    print("=" * 105)

    for r in table_rows:

        print(
            f"{r['System']:<17} "
            f"M={r['M']:<3} "
            f"Error={r['Error_Mean±Std']:<25} "
            f"TPR={r['TPR_Mean±Std']:<25} "
            f"TNR={r['TNR_Mean±Std']:<25} "
            f"Precision={r['Precision_Mean±Std']}"
        )

    # =================================================
    # 13. 打印三个系统M=10的VPS
    # =================================================

    print("\n" + "=" * 75)

    print(
        "M=10：三个系统的VPS（KN，论文Fig.8使用）"
    )

    print("=" * 75)

    if vps_kn_rows:

        for r in vps_kn_rows:

            print(
                f"{r['System']:<17} "
                f"VPS = {r['VPS_Mean']:.6f} "
                f"± {r['VPS_Std']:.6f}"
            )

    else:

        print(
            "未找到三个系统M=10的KN VPS结果。"
        )

    print("\nM=10：未知拓扑（UN）VPS，单独核对")

    for r in vps_rows:

        if r["Topology"] == "unknown":

            print(
                f"{r['System']:<17} "
                f"VPS = {r['VPS_Mean']:.6f} "
                f"± {r['VPS_Std']:.6f}"
            )

    # =================================================
    # 14. 最终检查
    # =================================================

    print("\n成功读取原始CSV文件数：", file_count)

    if missing:

        print("\n[警告] 以下预期文件缺失：")

        for p in missing:
            print("  ", p)

        print(
            "\n请补齐缺失实验，不要直接将不完整结果用于论文。"
        )

    else:

        print(
            "\n[检查通过] 所有预期文件均已找到。"
        )

    print("\n汇总文件保存位置：")
    print(output_dir)

    print("1. all_20runs_metrics_mean_std.csv")
    print("2. Table_II_unknown_topology.csv")
    print("3. VPS_M10_KN_for_Fig8.csv")
    print("4. VPS_M10_all_topologies.csv")


if __name__ == "__main__":

    main()
