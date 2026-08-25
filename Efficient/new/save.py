def save_cross_matrices(prediction_csv: Path, output_dir: Path, font_path: Optional[str]) -> None:
    df = pd.read_csv(prediction_csv).dropna(subset=["yolox_class_name", "eff_class_name"])
    if df.empty:
        print("[WARN] No classified detections; cross matrices were not generated.")
        return

    count = pd.crosstab(df["yolox_class_name"], df["eff_class_name"])
    ratio = pd.crosstab(
        df["yolox_class_name"],
        df["eff_class_name"],
        normalize="index"
    ) * 100

    yolox_order = [
        "1st stepladder",
        "2nd stepladder",
        "person N/A",
        "safe stepladder",
        "safe workplatform",
        "stepladder",
        "straddling",
        "unstable stepladder",
        "workplatform",
        "unstable workplatform",
    ]

    effnet_order = [
        "脚立1段目の人",
        "脚立2段目の人",
        "その他の人",
        "脚立上の安全な人",
        "立馬上の安全な人",
        "脚立",
        "脚立を跨ぐ人",
        "脚立上の不安定姿勢の人",
        "立馬",
        "立馬上の不安定姿勢の人",
    ]

    actual_yolox_names = {
        str(name).lower(): name
        for name in count.index
    }

    ordered_rows = [
        actual_yolox_names[name.lower()]
        for name in yolox_order
        if name.lower() in actual_yolox_names
    ]

    remaining_rows = [
        name
        for name in count.index
        if name not in ordered_rows
    ]

    ordered_columns = [
        name
        for name in effnet_order
        if name in count.columns
    ]

    remaining_columns = [
        name
        for name in count.columns
        if name not in ordered_columns
    ]

    final_row_order = ordered_rows + remaining_rows
    final_column_order = ordered_columns + remaining_columns

    count = count.reindex(
        index=final_row_order,
        columns=final_column_order,
        fill_value=0
    )

    ratio = ratio.reindex(
        index=final_row_order,
        columns=final_column_order,
        fill_value=0.0
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    count_csv = output_dir / "yolox_effnet_count_matrix.csv"
    ratio_csv = output_dir / "yolox_effnet_ratio_matrix.csv"
    count_png = output_dir / "yolox_effnet_count_heatmap.png"
    ratio_png = output_dir / "yolox_effnet_ratio_heatmap.png"

    count.to_csv(count_csv, encoding="utf-8-sig")
    ratio.to_csv(ratio_csv, encoding="utf-8-sig", float_format="%.2f")

    set_matrix_font(font_path)

    plot_matrix(
        count,
        count_png,
        "YOLOX-EfficientNetV2 Cross Matrix",
        False
    )

    plot_matrix(
        ratio,
        ratio_png,
        "YOLOX-EfficientNetV2 Cross Matrix (row normalized, %)",
        True
    )

    print("\n=== Cross matrix outputs ===")
    print(f"Count CSV   : {count_csv}")
    print(f"Ratio CSV   : {ratio_csv}")
    print(f"Count image : {count_png}")
    print(f"Ratio image : {ratio_png}")
    print("[INFO] These are prediction cross tables, not ground-truth confusion matrices.")