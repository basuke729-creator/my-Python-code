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

# =========================
# JSONアクセス用ロック管理
# =========================

_lock_file = None  # ロック状態を保持するための変数


def LockAppPref():
    """
    appPref.json にアクセスする前に呼ぶロック関数

    ・他の処理が使用中ならロック失敗（False）
    ・ロック成功したら True を返す

    ※ 必ず UnlockAppPref() とセットで使うこと
    """

    global _lock_file

    try:
        import msvcrt

        # すでにロック済みなら何もしない（多重ロック防止）
        if _lock_file is not None:
            return True

        # ロック用ファイルを開く（なければ作成される）
        _lock_file = open("appPref.lock", "w")

        # 先頭1バイトをロックする（非ブロッキング）
        # → 他で使われていたら即失敗する
        msvcrt.locking(_lock_file.fileno(), msvcrt.LK_NBLCK, 1)

        return True  # ロック成功

    except Exception as e:
        # ロック失敗（他プロセスが使用中など）
        _lock_file = None
        return False


def UnlockAppPref():
    """
    LockAppPrefで取得したロックを解除する

    ※ LockAppPref成功時のみ呼ぶこと
    """

    global _lock_file

    try:
        import msvcrt

        # ロックしている場合のみ解除する
        if _lock_file is not None:

            # ロック解除
            msvcrt.locking(_lock_file.fileno(), msvcrt.LK_UNLCK, 1)

            # ファイルを閉じる
            _lock_file.close()

            # 状態をリセット
            _lock_file = None

    except Exception as e:
        # エラーが出ても止めない（安全側）
        pass


using System;
using System.IO;

public static class AppPrefLock
{
    // ロック用のファイルストリームを保持
    private static FileStream _lockFile = null;

    /// <summary>
    /// appPref.json 用のロックを取得する
    /// 成功: true
    /// 失敗: false
    /// </summary>
    public static bool LockAppPref()
    {
        try
        {
            // すでにロックしている場合はそのままOK
            if (_lockFile != null)
            {
                return true;
            }

            // ロック用ファイルを開く（なければ作成）
            // FileShare.None にすることで「他から開けない」状態にする
            _lockFile = new FileStream(
                "appPref.lock",
                FileMode.OpenOrCreate,
                FileAccess.ReadWrite,
                FileShare.None
            );

            return true; // ロック成功
        }
        catch (IOException)
        {
            // 他のプロセスがすでにロックしている場合
            _lockFile = null;
            return false;
        }
        catch (Exception)
        {
            _lockFile = null;
            return false;
        }
    }

    /// <summary>
    /// ロックを解除する
    /// </summary>
    public static void UnlockAppPref()
    {
        try
        {
            if (_lockFile != null)
            {
                // ファイルを閉じるとロック解除される
                _lockFile.Close();
                _lockFile = null;
            }
        }
        catch (Exception)
        {
            // エラーは無視（安全側）
        }
    }
}

import json
import os
import time
import threading

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


# =========================
# 設定
# =========================
APP_PREF_PATH = "appPref.json"


# =========================
# グローバル変数
# =========================
_app_pref_data = {}          # JSONの最新内容を保持する
_app_pref_lock = threading.Lock()  # スレッド安全のためのロック
_observer = None             # watchdog の Observer を保持


# =========================
# JSON読込処理
# =========================
def _load_app_pref():
    """
    appPref.json を読み込んで、グローバル変数に反映する内部関数
    読み込み失敗時は何もしない
    """
    global _app_pref_data

    try:
        with open(APP_PREF_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 読み込み成功したら置き換える
        with _app_pref_lock:
            _app_pref_data = data

        print("appPref.json を再読込しました")

    except Exception as e:
        # C# 側が書き込み途中の瞬間などは失敗することがある
        print(f"appPref.json の読込に失敗しました: {e}")


# =========================
# watchdog用イベントハンドラ
# =========================
class AppPrefEventHandler(FileSystemEventHandler):
    """
    appPref.json の更新を検知するためのクラス
    """

    def on_modified(self, event):
        # フォルダ変更は無視
        if event.is_directory:
            return

        # 変更されたファイル名だけ取り出す
        changed_file = os.path.basename(event.src_path)

        # 対象が appPref.json のときだけ再読込
        if changed_file == os.path.basename(APP_PREF_PATH):
            print("appPref.json の更新を検知しました")
            _load_app_pref()

    def on_created(self, event):
        # ファイルが作り直された場合にも対応
        if event.is_directory:
            return

        changed_file = os.path.basename(event.src_path)

        if changed_file == os.path.basename(APP_PREF_PATH):
            print("appPref.json の作成を検知しました")
            _load_app_pref()


# =========================
# 監視開始関数
# =========================
def StartAppPrefWatch():
    """
    appPref.json の監視を開始する
    最初に1回読み込んでから watchdog を開始する
    """
    global _observer

    if _observer is not None:
        print("すでに監視中です")
        return

    # 最初に1回読み込む
    _load_app_pref()

    # appPref.json があるフォルダを監視する
    watch_dir = os.path.dirname(os.path.abspath(APP_PREF_PATH))
    event_handler = AppPrefEventHandler()

    observer = Observer()
    observer.schedule(event_handler, watch_dir, recursive=False)
    observer.start()

    _observer = observer
    print("appPref.json の監視を開始しました")


# =========================
# 監視停止関数
# =========================
def StopAppPrefWatch():
    """
    appPref.json の監視を停止する
    """
    global _observer

    if _observer is not None:
        _observer.stop()
        _observer.join()
        _observer = None
        print("appPref.json の監視を停止しました")


# =========================
# 外から使う読出し関数
# =========================
def GetAppPref():
    """
    appPref.json の最新内容を辞書型で返す
    引数なし
    """
    with _app_pref_lock:
        return dict(_app_pref_data)