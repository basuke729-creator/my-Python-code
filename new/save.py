def save_cross_matrices(output_csv, matrix_dir, matrix_font=None):
    """
    YOLOXの検出クラスとEfficientNetV2の分類クラスから
    クロス集計表とヒートマップを作成する。

    注意:
        これは正解ラベルを使用した混同行列ではなく、
        YOLOX予測 × EfficientNetV2予測のクロス集計である。

    Parameters
    ----------
    output_csv : str or pathlib.Path
        推論結果CSVのパス

    matrix_dir : str or pathlib.Path
        クロス集計結果の保存先ディレクトリ

    matrix_font : str or pathlib.Path or None
        日本語フォントファイルのパス
    """

    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib import font_manager


    # ============================================================
    # 入出力パス
    # ============================================================
    output_csv = Path(output_csv)
    matrix_dir = Path(matrix_dir)

    matrix_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not output_csv.exists():
        print(f"[WARN] CSV file not found: {output_csv}")
        return


    # ============================================================
    # CSV読み込み
    # ============================================================
    try:
        df = pd.read_csv(output_csv)
    except Exception as exc:
        print(f"[WARN] Failed to read CSV: {output_csv}")
        print(f"[WARN] {exc}")
        return

    if df.empty:
        print("[WARN] Prediction CSV is empty.")
        print("[WARN] Cross matrices were not generated.")
        return


    # ============================================================
    # YOLOX列・EfficientNetV2列を探す
    #
    # 現在のCSVに合わせて、候補名を複数用意している。
    # ============================================================
    yolox_column_candidates = [
        "yolox_class_name",
        "yolox_class",
        "yolo_class_name",
        "yolo_class",
        "yolox_cls_name",
        "yolo_cls_name",
        "yolox_cls",
        "yolo_cls",
    ]

    effnet_column_candidates = [
        "effnet_class_name",
        "effnet_class",
        "eff_class_name",
        "eff_class",
        "efficientnet_class_name",
        "efficientnet_class",
        "effnet_cls_name",
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
        print("[WARN] YOLOX or EfficientNetV2 class column was not found.")
        print(f"[INFO] CSV columns: {list(df.columns)}")
        print(
            "[INFO] Edit yolox_column_candidates and "
            "effnet_column_candidates if necessary."
        )
        return


    # ============================================================
    # 欠損データを除外
    # ============================================================
    matrix_df = df[
        [yolox_column, effnet_column]
    ].dropna().copy()

    if matrix_df.empty:
        print("[WARN] No valid YOLOX/EfficientNetV2 predictions found.")
        return

    # クラス名を文字列に統一し、余分な空白を除去
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


    # ============================================================
    # 表示順
    #
    # ユーザー指定のYOLOX側の並びに合わせて、
    # EfficientNetV2側も意味が対応する順番にする。
    # ============================================================
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


    # ============================================================
    # YOLOXクラス名の大文字・小文字差を吸収
    #
    # 実際のCSVが
    #   Person N/A
    #   Safe stepladder
    # のような表記でも、指定順に並ぶようにする。
    # ============================================================
    yolox_name_lookup = {
        str(name).strip().lower(): str(name).strip()
        for name in matrix_df[yolox_column].unique()
    }

    resolved_yolox_order = []

    for requested_name in yolox_order:
        actual_name = yolox_name_lookup.get(
            requested_name.lower()
        )

        if actual_name is not None:
            resolved_yolox_order.append(actual_name)
        else:
            # 動画中に出現していない場合も、指定表記で行を残す
            resolved_yolox_order.append(requested_name)


    # ============================================================
    # 件数クロス集計
    # ============================================================
    count_matrix = pd.crosstab(
        matrix_df[yolox_column],
        matrix_df[effnet_column],
        dropna=False,
    )


    # ============================================================
    # 指定外のクラスが存在した場合、最後尾へ残す
    # ============================================================
    extra_yolox_classes = [
        class_name
        for class_name in count_matrix.index
        if class_name not in resolved_yolox_order
    ]

    extra_effnet_classes = [
        class_name
        for class_name in count_matrix.columns
        if class_name not in effnet_order
    ]

    final_yolox_order = (
        resolved_yolox_order
        + extra_yolox_classes
    )

    final_effnet_order = (
        effnet_order
        + extra_effnet_classes
    )


    # ============================================================
    # 件数行列を並べ替え
    #
    # 出現しなかったクラスは0で補完する。
    # ============================================================
    count_matrix = count_matrix.reindex(
        index=final_yolox_order,
        columns=final_effnet_order,
        fill_value=0,
    )


    # ============================================================
    # 行ごとの割合行列を作成
    #
    # 各YOLOXクラスについて、
    # EfficientNetV2がどのクラスへ分類したかを%で表す。
    # ============================================================
    row_totals = count_matrix.sum(axis=1)

    ratio_matrix = count_matrix.div(
        row_totals.replace(0, np.nan),
        axis=0,
    ) * 100.0

    # 該当データがない行は0.0%
    ratio_matrix = ratio_matrix.fillna(0.0)


    # ============================================================
    # 出力パス
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
    # ============================================================
    font_properties = None

    if matrix_font:
        matrix_font = Path(matrix_font)

        if matrix_font.exists():
            try:
                font_properties = (
                    font_manager.FontProperties(
                        fname=str(matrix_font)
                    )
                )
            except Exception as exc:
                print(
                    "[WARN] Failed to load matrix font: "
                    f"{matrix_font}"
                )
                print(f"[WARN] {exc}")
        else:
            print(
                "[WARN] Matrix font file not found: "
                f"{matrix_font}"
            )


    # ============================================================
    # ヒートマップ描画関数
    # ============================================================
    def plot_matrix(
        matrix,
        output_path,
        title,
        value_format,
        color_max=None,
    ):
        rows, columns = matrix.shape

        # クラス数に合わせて画像サイズを調整
        figure_width = max(
            12.0,
            columns * 1.25,
        )

        figure_height = max(
            8.0,
            rows * 0.75,
        )

        fig, ax = plt.subplots(
            figsize=(figure_width, figure_height)
        )

        values = matrix.to_numpy(
            dtype=float
        )

        if color_max is None:
            image = ax.imshow(
                values,
                aspect="auto",
                cmap="Blues",
                interpolation="nearest",
            )
        else:
            image = ax.imshow(
                values,
                aspect="auto",
                cmap="Blues",
                interpolation="nearest",
                vmin=0,
                vmax=color_max,
            )

        colorbar = fig.colorbar(
            image,
            ax=ax,
            fraction=0.046,
            pad=0.04,
        )

        if color_max == 100:
            colorbar.set_label(
                "Percentage (%)",
                fontproperties=font_properties,
            )
        else:
            colorbar.set_label(
                "Count",
                fontproperties=font_properties,
            )


        # 軸ラベル
        ax.set_xticks(
            np.arange(columns)
        )

        ax.set_yticks(
            np.arange(rows)
        )

        ax.set_xticklabels(
            matrix.columns,
            rotation=55,
            ha="right",
            rotation_mode="anchor",
            fontproperties=font_properties,
        )

        ax.set_yticklabels(
            matrix.index,
            fontproperties=font_properties,
        )

        ax.set_xlabel(
            "EfficientNetV2 classification",
            fontproperties=font_properties,
            fontsize=13,
        )

        ax.set_ylabel(
            "YOLOX detection",
            fontproperties=font_properties,
            fontsize=13,
        )

        ax.set_title(
            title,
            fontproperties=font_properties,
            fontsize=15,
            pad=14,
        )


        # ========================================================
        # セル内に数値を表示
        # ========================================================
        if values.size > 0:
            display_max = float(
                np.nanmax(values)
            )
        else:
            display_max = 0.0

        threshold = (
            display_max * 0.5
            if display_max > 0
            else 0.0
        )

        for row_index in range(rows):
            for column_index in range(columns):
                value = values[
                    row_index,
                    column_index,
                ]

                if value_format == "count":
                    display_text = f"{int(round(value))}"
                else:
                    display_text = f"{value:.1f}"

                text_color = (
                    "white"
                    if value > threshold
                    else "black"
                )

                ax.text(
                    column_index,
                    row_index,
                    display_text,
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=9,
                    fontproperties=font_properties,
                )


        # セル境界を見やすくする
        ax.set_xticks(
            np.arange(-0.5, columns, 1),
            minor=True,
        )

        ax.set_yticks(
            np.arange(-0.5, rows, 1),
            minor=True,
        )

        ax.grid(
            which="minor",
            color="white",
            linestyle="-",
            linewidth=0.7,
        )

        ax.tick_params(
            which="minor",
            bottom=False,
            left=False,
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
            "YOLOX–EfficientNetV2 Cross Matrix "
            "(count)"
        ),
        value_format="count",
        color_max=None,
    )


    # ============================================================
    # 割合ヒートマップ
    # ============================================================
    plot_matrix(
        matrix=ratio_matrix,
        output_path=ratio_png_path,
        title=(
            "YOLOX–EfficientNetV2 Cross Matrix "
            "(row normalized, %)"
        ),
        value_format="ratio",
        color_max=100,
    )


    # ============================================================
    # 出力結果表示
    # ============================================================
    print("\n=== Cross matrix outputs ===")
    print(f"Count CSV    : {count_csv_path}")
    print(f"Ratio CSV    : {ratio_csv_path}")
    print(f"Count image  : {count_png_path}")
    print(f"Ratio image  : {ratio_png_path}")

    print(
        "\n[INFO] These are prediction cross tables, "
        "not ground-truth confusion matrices."
    )