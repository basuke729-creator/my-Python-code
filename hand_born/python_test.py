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
# 仕様:
# - update_pref は辞書型
# - 全読出し
# - 一部更新
# - 全書込み
# - ロック付き
# =========================================================
def SetAppPref(update_pref: dict) -> bool:
    global _app_pref_fp

    if not isinstance(update_pref, dict):
        return False

    if not LockAppPref():
        return False

    try:
        # ロック済みの同じファイルポインタで読む
        _app_pref_fp.seek(0)
        text = _app_pref_fp.read()

        if text.strip() == "":
            data = {}
        else:
            data = json.loads(text)

        # 一部更新
        for key, value in update_pref.items():
            if key in data and isinstance(data[key], dict):
                data[key]["Value"] = str(value)

        # 全書込み
        _app_pref_fp.seek(0)
        _app_pref_fp.truncate()
        json.dump(data, _app_pref_fp, ensure_ascii=False, indent=4)
        _app_pref_fp.flush()
        os.fsync(_app_pref_fp.fileno())

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
# 仕様:
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
        # ロック済みの同じファイルポインタで読む
        _app_pref_fp.seek(0)
        text = _app_pref_fp.read()

        if text.strip() == "":
            return {}

        data = json.loads(text)

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
# 補助: 現在値を直接読む
# =========================================================
def direct_read_mode_and_camera_status():
    global _app_pref_fp

    if not LockAppPref():
        print("Python: Lock失敗")
        return

    try:
        _app_pref_fp.seek(0)
        text = _app_pref_fp.read()

        if text.strip() == "":
            print("Python: JSONが空です")
            return

        data = json.loads(text)

        print("Mode =", data.get("Mode", {}).get("Value"))
        print("CameraStatus =", data.get("CameraStatus", {}).get("Value"))

    except Exception as e:
        print("Python: direct read error:", e)

    finally:
        UnlockAppPref()


# =========================================================
# テスト用
# =========================================================
def test_python_write():
    print("Python: Mode=2, CameraStatus=0 に更新します")
    ok = SetAppPref({
        "Mode": "2",
        "CameraStatus": "0"
    })
    print("Python: SetAppPref result =", ok)


def test_python_restore():
    print("Python: Mode=1, CameraStatus=1 に戻します")
    ok = SetAppPref({
        "Mode": "1",
        "CameraStatus": "1"
    })
    print("Python: Restore result =", ok)


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
        print("Mode =", prefs.get("Mode"))
        print("CameraStatus =", prefs.get("CameraStatus"))
    else:
        print("Python: 読み取り失敗またはタイムアウト")


if __name__ == "__main__":
    print("=== Python Production JSON Test Menu ===")
    print("1: Python から Mode=2, CameraStatus=0 に更新")
    print("2: Python でロックして10秒保持")
    print("3: ロックできるか試すだけ")
    print("4: C#更新待ちして Mode, CameraStatus を読む")
    print("5: 今のJSONから Mode, CameraStatus を直接表示")
    print("6: Mode=1, CameraStatus=1 に戻す")
    choice = input("番号を入力してください: ").strip()

    if choice == "1":
        test_python_write()
    elif choice == "2":
        test_python_lock_hold()
    elif choice == "3":
        test_python_try_lock_only()
    elif choice == "4":
        test_wait_csharp_update_and_read()
    elif choice == "5":
        direct_read_mode_and_camera_status()
    elif choice == "6":
        test_python_restore()
    else:
        print("不正な入力です")