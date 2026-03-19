import json
import os
from typing import Dict, Any


APP_PREFS_JSON_PATH = "AppPrefs.json"


def SetAppPref(update_pref: Dict[str, Any], json_path: str = APP_PREFS_JSON_PATH) -> bool:
    """
    libAdamApiPython.adam_set_appPref の代替用関数

    Parameters
    ----------
    update_pref : dict
        更新したい設定値
        例:
        {
            "CameraStatus": 0,
            "Mode": 1,
            "ROIWidth": 0.5,
            "ModelName": "sample_model"
        }

    json_path : str
        AppPrefs.json のパス

    Returns
    -------
    bool
        更新成功時 True
    """

    if not isinstance(update_pref, dict):
        raise TypeError("update_pref は dict である必要があります")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSONファイルが存在しません: {json_path}")

    # 全読み込み
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 一部更新
    for key, new_value in update_pref.items():
        if key not in data:
            print(f"[WARNING] キーが存在しません: {key}")
            continue

        if not isinstance(data[key], dict):
            print(f"[WARNING] {key} の構造が想定と異なります")
            continue

        if "Value" not in data[key]:
            print(f"[WARNING] {key} に 'Value' がありません")
            continue

        # AppPrefs.json の見た目に合わせて文字列で保存
        data[key]["Value"] = str(new_value)

    # 全書き戻し
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    return True