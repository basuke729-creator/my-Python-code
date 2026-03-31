import json
import os
import time
import msvcrt

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


# =========================================================
# 本番JSONパス
# C#側と必ず同じにしてください
# =========================================================
APP_PREF_PATH = r"C:\i-pro\27_PC版対応\00_調査\jsonsample\AppPrefs.json"

_app_pref_fp = None
_app_pref_lock_size = 0


# =========================================================
# アクセス禁止関数
# =========================================================
def LockAppPref() -> bool:
    global _app_pref_fp
    global _app_pref_lock_size

    try:
        if _app_pref_fp is not None:
            return False

        if not os.path.exists(APP_PREF_PATH):
            print("LockAppPref: JSONファイルが存在しません")
            return False

        _app_pref_fp = open(APP_PREF_PATH, "r+", encoding="utf-8")

        _app_pref_fp.seek(0, os.SEEK_END)
        _app_pref_lock_size = _app_pref_fp.tell()

        if _app_pref_lock_size <= 0:
            _app_pref_lock_size = 1

        _app_pref_fp.seek(0)
        msvcrt.locking(_app_pref_fp.fileno(), msvcrt.LK_NBLCK, _app_pref_lock_size)

        return True

    except Exception as e:
        print("LockAppPref error:", e)

        if _app_pref_fp is not None:
            try:
                _app_pref_fp.close()
            except:
                pass

        _app_pref_fp = None
        _app_pref_lock_size = 0
        return False


# =========================================================
# アクセス許可関数
# =========================================================
def UnlockAppPref() -> None:
    global _app_pref_fp
    global _app_pref_lock_size

    try:
        if _app_pref_fp is not None:
            _app_pref_fp.seek(0)

            unlock_size = _app_pref_lock_size
            if unlock_size <= 0:
                unlock_size = 1

            msvcrt.locking(_app_pref_fp.fileno(), msvcrt.LK_UNLCK, unlock_size)
            _app_pref_fp.close()
    except Exception as e:
        print("UnlockAppPref error:", e)
    finally:
        _app_pref_fp = None
        _app_pref_lock_size = 0


# =========================================================
# json更新用関数
# update_pref の key に対応する項目の Value を更新
# =========================================================
def SetAppPref(update_pref: dict) -> bool:
    if not isinstance(update_pref, dict):
        return False

    if not LockAppPref():
        return False

    try:
        with open(APP_PREF_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        for key, value in update_pref.items():
            if key in data and isinstance(data[key], dict):
                data[key]["Value"] = str(value)

        with open(APP_PREF_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        return True

    except Exception as e:
        print("SetAppPref error:", e)
        return False

    finally:
        UnlockAppPref()


# =========================================================
# watchdog用
# =========================================================
class _JsonUpdateHandler(FileSystemEventHandler):
    def __init__(self):
        self.updated = False

    def on_modified(self, event):
        try:
            event_path = os.path.abspath(event.src_path)
            target_path = os.path.abspath(APP_PREF_PATH)

            if event_path == target_path:
                self.updated = True
        except:
            pass


# =========================================================
# json読出し用関数
# 要件：
# - 引数なし
# - C#からjsonが更新されたことを検知して読出し
# - 全パラメータの値を辞書型で返す
# =========================================================
def GetAppPref() -> dict:
    watch_dir = os.path.dirname(APP_PREF_PATH)
    if watch_dir == "":
        watch_dir = "."

    handler = _JsonUpdateHandler()
    observer = Observer()
    observer.schedule(handler, path=watch_dir, recursive=False)
    observer.start()

    try:
        start_time = time.time()
        timeout_sec = 30

        while True:
            if handler.updated:
                break

            if time.time() - start_time > timeout_sec:
                print("GetAppPref: 更新待ちタイムアウト")
                return {}

            time.sleep(0.1)

    finally:
        observer.stop()
        observer.join()

    if not LockAppPref():
        print("GetAppPref: Lock失敗")
        return {}

    try:
        with open(APP_PREF_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        result = {}
        for key, value in data.items():
            if isinstance(value, dict):
                result[key] = value.get("Value", None)
            else:
                result[key] = None

        return result

    except Exception as e:
        print("GetAppPref error:", e)
        return {}

    finally:
        UnlockAppPref()


# =========================================================
# テスト用関数
# 本番JSONでは比較的安全なキーを使う
# =========================================================
def test_python_write_production_json():
    print("Python: ModelName, WorkProcedureName を更新します")
    ok = SetAppPref({
        "ModelName": "PYTHON_TEST_MODEL",
        "WorkProcedureName": "PYTHON_TEST_WORK"
    })
    print("Python: SetAppPref result =", ok)


def test_python_lock_hold():
    print("Python: LockAppPref 開始")
    ok = LockAppPref()
    print("Python: Lock result =", ok)

    if ok:
        try:
            print("Python: 10秒ロック保持します。今のうちにC#でロックを試してください。")
            time.sleep(10)
        finally:
            UnlockAppPref()
            print("Python: Unlock 完了")


def test_python_try_lock_only():
    print("Python: LockAppPref 試行")
    ok = LockAppPref()
    print("Python: Lock result =", ok)

    if ok:
        UnlockAppPref()
        print("Python: すぐUnlockしました")


def test_wait_csharp_update_and_read():
    print("Python: C#からの更新待ちを開始します")
    prefs = GetAppPref()

    if prefs:
        print("Python: Wait and Read result")
        print("ModelName =", prefs.get("ModelName"))
        print("WorkProcedureName =", prefs.get("WorkProcedureName"))
    else:
        print("Python: 読み取り失敗またはタイムアウト")


def test_read_current_json_direct():
    print("Python: 現在のJSONを直接読みます")

    if not LockAppPref():
        print("Python: Lock失敗")
        return

    try:
        with open(APP_PREF_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        print("ModelName =", data.get("ModelName", {}).get("Value"))
        print("WorkProcedureName =", data.get("WorkProcedureName", {}).get("Value"))
    except Exception as e:
        print("Python: direct read error:", e)
    finally:
        UnlockAppPref()


if __name__ == "__main__":
    print("=== Python Production JSON Test Menu ===")
    print("1: Python から ModelName, WorkProcedureName を更新")
    print("2: Python でロックして10秒保持")
    print("3: ロックできるか試すだけ")
    print("4: C#更新待ちして全件読込む")
    print("5: 今のJSONから ModelName, WorkProcedureName を直接表示")
    choice = input("番号を入力してください: ").strip()

    if choice == "1":
        test_python_write_production_json()
    elif choice == "2":
        test_python_lock_hold()
    elif choice == "3":
        test_python_try_lock_only()
    elif choice == "4":
        test_wait_csharp_update_and_read()
    elif choice == "5":
        test_read_current_json_direct()
    else:
        print("不正な入力です")