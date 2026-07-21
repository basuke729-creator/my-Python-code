def save_cross_matrices(output_csv, matrix_dir, matrix_font=None):
    """
    YOLOXの検出結果とEfficientNetV2の分類結果をクロス集計し、
    件数・行方向割合のCSVとヒートマップを保存する。

    ※正解ラベルを使用した混同行列ではなく、
      YOLOX予測 × EfficientNetV2予測のクロス集計。
    """

    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib import font_manager


    # ============================================================
    # パス設定
    # ============================================================
    output_csv = Path(output_csv)
    matrix_dir = Path(matrix_dir)

    matrix_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not output_csv.exists():
        print(f"[WARN] CSV not found: {output_csv}")
        return


    # ============================================================
    # 推論結果CSVを読み込む
    # ============================================================
    df = pd.read_csv(output_csv)

    if df.empty:
        print("[WARN] Prediction CSV is empty.")
        return


    # ============================================================
    # CSV内のクラス名列を取得
    #
    # 現在のCSVの列名に応じて候補から検索する
    # ============================================================
    yolox_column_candidates = [
        "yolox_class",
        "yolox_class_name",
        "yolo_class",
        "yolo_class_name",
        "yolox_cls",
        "yolo_cls",
    ]

    effnet_column_candidates = [
        "eff_class",
        "eff_class_name",
        "effnet_class",
        "effnet_class_name",
        "efficientnet_class",
        "efficientnet_class_name",
        "effnet_cls",
    ]

    yolox_column = next(
        (
            column
            for column in yolox_column_candidates
            if column in df.columns
        ),
        None,
    )

    effnet_column = next(
        (
            column
            for column in effnet_column_candidates
            if column in df.columns
        ),
        None,
    )

    if yolox_column is None or effnet_column is None:
        print("[WARN] Class-name columns were not found.")
        print(f"[INFO] CSV columns: {list(df.columns)}")
        return


    # ============================================================
    # クラス名を文字列に統一
    # ============================================================
    matrix_df = df[
        [yolox_column, effnet_column]
    ].dropna().copy()

    matrix_df[yolox_column] = (
        matrix_df[yolox_column]
        .astype(str)
        .str.strip()
    )

    matrix_df[effnet_column] = (
        matrix_df[effnet_column]
        .astype(str)
        .str.strip()
    )

    if matrix_df.empty:
        print("[WARN] No valid class data found.")
        return


    # ============================================================
    # 件数クロス集計を作成
    # ============================================================
    count_matrix = pd.crosstab(
        matrix_df[yolox_column],
        matrix_df[effnet_column],
        dropna=False,
    )


    # ============================================================
    # 行方向の割合を作成
    # ============================================================
    ratio_matrix = count_matrix.div(
        count_matrix.sum(axis=1).replace(0, np.nan),
        axis=0,
    ) * 100.0

    ratio_matrix = ratio_matrix.fillna(0.0)


    # ============================================================
    # 今回追加するクラス順の並び替え
    # ============================================================

    # 縦軸：YOLOX
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

    # 横軸：EfficientNetV2
    # YOLOX側と意味が対応する順番
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


    # ------------------------------------------------------------
    # YOLOX側：
    # 大文字・小文字の違いを吸収して実データの名前を取得する
    # ------------------------------------------------------------
    actual_yolox_names = {
        str(name).strip().lower(): str(name).strip()
        for name in count_matrix.index
    }

    ordered_rows = []

    for requested_name in yolox_order:
        actual_name = actual_yolox_names.get(
            requested_name.lower()
        )

        if actual_name is not None:
            ordered_rows.append(actual_name)


    # 指定リストにないYOLOXクラスがあれば末尾に残す
    remaining_rows = [
        name
        for name in count_matrix.index
        if name not in ordered_rows
    ]

    final_row_order = ordered_rows + remaining_rows


    # ------------------------------------------------------------
    # EfficientNetV2側：
    # 実際に存在する列を指定順に並べる
    # ------------------------------------------------------------
    ordered_columns = [
        name
        for name in effnet_order
        if name in count_matrix.columns
    ]


    # 指定リストにないEfficientNetV2クラスがあれば末尾に残す
    remaining_columns = [
        name
        for name in count_matrix.columns
        if name not in ordered_columns
    ]

    final_column_order = (
        ordered_columns
        + remaining_columns
    )


    # ------------------------------------------------------------
    # 件数行列と割合行列を同じ順番に並べ替える
    # ------------------------------------------------------------
    count_matrix = count_matrix.reindex(
        index=final_row_order,
        columns=final_column_order,
        fill_value=0,
    )

    ratio_matrix = ratio_matrix.reindex(
        index=final_row_order,
        columns=final_column_order,
        fill_value=0.0,
    )


    # ============================================================
    # 保存先
    # ============================================================
    count_csv_path = (
        matrix_dir
        / "yolox_effnet_count_matrix.csv"
    )

    ratio_csv_path = (
        matrix_dir
        / "yolox_effnet_ratio_matrix.csv"
    )

    count_png_path = (
        matrix_dir
        / "yolox_effnet_count_heatmap.png"
    )

    ratio_png_path = (
        matrix_dir
        / "yolox_effnet_ratio_heatmap.png"
    )


    # ============================================================
    # CSV保存
    # ============================================================
    count_matrix.to_csv(
        count_csv_path,
        encoding="utf-8-sig",
    )

    ratio_matrix.to_csv(
        ratio_csv_path,
        encoding="utf-8-sig",
        float_format="%.3f",
    )


    # ============================================================
    # 日本語フォント設定
    #
    # 修正前に使用していた --matrix-font の指定をそのまま使用
    # ============================================================
    font_properties = None

    if matrix_font:
        matrix_font = Path(matrix_font)

        if matrix_font.exists():
            font_properties = font_manager.FontProperties(
                fname=str(matrix_font)
            )
        else:
            print(
                f"[WARN] Font file not found: {matrix_font}"
            )


    # ============================================================
    # ヒートマップ描画
    # ============================================================
    def plot_matrix(
        matrix,
        output_path,
        title,
        is_ratio=False,
    ):
        row_count, column_count = matrix.shape

        fig_width = max(
            12,
            column_count * 1.2,
        )

        fig_height = max(
            8,
            row_count * 0.75,
        )

        fig, ax = plt.subplots(
            figsize=(fig_width, fig_height)
        )

        values = matrix.to_numpy(
            dtype=float
        )

        if is_ratio:
            image = ax.imshow(
                values,
                cmap="Blues",
                aspect="auto",
                interpolation="nearest",
                vmin=0,
                vmax=100,
            )
        else:
            image = ax.imshow(
                values,
                cmap="Blues",
                aspect="auto",
                interpolation="nearest",
            )


        # カラーバー
        colorbar = fig.colorbar(
            image,
            ax=ax,
        )

        if is_ratio:
            colorbar.set_label("Percentage (%)")
        else:
            colorbar.set_label("Count")


        # 軸目盛り
        ax.set_xticks(
            np.arange(column_count)
        )

        ax.set_yticks(
            np.arange(row_count)
        )


        # 横軸ラベル
        ax.set_xticklabels(
            matrix.columns,
            rotation=55,
            ha="right",
            rotation_mode="anchor",
            fontproperties=font_properties,
        )


        # 縦軸ラベル
        ax.set_yticklabels(
            matrix.index,
            fontproperties=font_properties,
        )


        ax.set_xlabel(
            "EfficientNetV2 classification"
        )

        ax.set_ylabel(
            "YOLOX detection"
        )

        ax.set_title(
            title
        )


        # ========================================================
        # セル内に値を表示
        # ========================================================
        if values.size > 0:
            maximum_value = float(
                np.nanmax(values)
            )
        else:
            maximum_value = 0.0

        threshold = maximum_value / 2.0

        for row_index in range(row_count):
            for column_index in range(column_count):
                value = values[
                    row_index,
                    column_index
                ]

                if is_ratio:
                    text = f"{value:.1f}"
                else:
                    text = f"{int(value)}"

                text_color = (
                    "white"
                    if value > threshold
                    else "black"
                )

                ax.text(
                    column_index,
                    row_index,
                    text,
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=9,
                    fontproperties=font_properties,
                )


        fig.tight_layout()

        fig.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight",
        )

        plt.close(fig)


    # ============================================================
    # 件数ヒートマップ
    # ============================================================
    plot_matrix(
        matrix=count_matrix,
        output_path=count_png_path,
        title=(
            "YOLOX–EfficientNetV2 "
            "Cross Matrix (count)"
        ),
        is_ratio=False,
    )


    # ============================================================
    # 割合ヒートマップ
    # ============================================================
    plot_matrix(
        matrix=ratio_matrix,
        output_path=ratio_png_path,
        title=(
            "YOLOX–EfficientNetV2 "
            "Cross Matrix (row normalized, %)"
        ),
        is_ratio=True,
    )


    # ============================================================
    # 出力結果
    # ============================================================
    print("\n=== Cross matrix outputs ===")
    print(f"Count CSV   : {count_csv_path}")
    print(f"Ratio CSV   : {ratio_csv_path}")
    print(f"Count image : {count_png_path}")
    print(f"Ratio image : {ratio_png_path}")

    print(
        "\n[INFO] These are prediction cross tables, "
        "not ground-truth confusion matrices."
    )