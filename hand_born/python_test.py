import json
import os
import time
import msvcrt

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

APP_PREF_PATH = r"C:\temp\AppPrefs.json"
_app_pref_fp = None


def LockAppPref() -> bool:
    global _app_pref_fp

    try:
        if _app_pref_fp is not None:
            return False

        if not os.path.exists(APP_PREF_PATH):
            with open(APP_PREF_PATH, "w", encoding="utf-8") as f:
                json.dump({}, f, ensure_ascii=False, indent=4)

        _app_pref_fp = open(APP_PREF_PATH, "r+", encoding="utf-8")
        _app_pref_fp.seek(0)
        msvcrt.locking(_app_pref_fp.fileno(), msvcrt.LK_NBLCK, 1)
        return True

    except Exception as e:
        print("LockAppPref error:", e)
        if _app_pref_fp is not None:
            try:
                _app_pref_fp.close()
            except:
                pass
            _app_pref_fp = None
        return False


def UnlockAppPref() -> None:
    global _app_pref_fp

    try:
        if _app_pref_fp is not None:
            _app_pref_fp.seek(0)
            msvcrt.locking(_app_pref_fp.fileno(), msvcrt.LK_UNLCK, 1)
            _app_pref_fp.close()
    except Exception as e:
        print("UnlockAppPref error:", e)
    finally:
        _app_pref_fp = None


def GetAppPref(data_name: str):
    global _app_pref_fp

    if _app_pref_fp is None:
        return None

    try:
        _app_pref_fp.seek(0)
        text = _app_pref_fp.read()

        if text.strip() == "":
            return None

        data = json.loads(text)

        if data_name not in data:
            return None

        if isinstance(data[data_name], dict) and "Value" in data[data_name]:
            return data[data_name]["Value"]

        return None

    except Exception as e:
        print("GetAppPref error:", e)
        return None


def SetAppPref(update_pref: dict) -> bool:
    global _app_pref_fp

    if not isinstance(update_pref, dict):
        return False

    if _app_pref_fp is None:
        return False

    try:
        _app_pref_fp.seek(0)
        text = _app_pref_fp.read()

        if text.strip() == "":
            data = {}
        else:
            data = json.loads(text)

        for key, value in update_pref.items():
            value_str = str(value)

            if key in data and isinstance(data[key], dict):
                data[key]["Value"] = value_str
            else:
                data[key] = {
                    "PrefType": "String",
                    "CameraAccess": "ReadWrite",
                    "ScreenAccess": "ReadWrite",
                    "Value": value_str
                }

        _app_pref_fp.seek(0)
        _app_pref_fp.truncate()
        json.dump(data, _app_pref_fp, ensure_ascii=False, indent=4)
        _app_pref_fp.flush()
        os.fsync(_app_pref_fp.fileno())

        return True

    except Exception as e:
        print("SetAppPref error:", e)
        return False


def WaitAppPrefUpdated(timeout_sec=None) -> bool:
    watch_dir = os.path.dirname(APP_PREF_PATH)
    if watch_dir == "":
        watch_dir = "."

    updated = {"done": False}

    class JsonChangeHandler(FileSystemEventHandler):
        def on_modified(self, event):
            if event.is_directory:
                return

            event_path = os.path.abspath(event.src_path)
            target_path = os.path.abspath(APP_PREF_PATH)

            if event_path == target_path:
                updated["done"] = True

    observer = Observer()
    handler = JsonChangeHandler()
    observer.schedule(handler, watch_dir, recursive=False)
    observer.start()

    start_time = time.time()

    try:
        while not updated["done"]:
            time.sleep(0.1)

            if timeout_sec is not None:
                if (time.time() - start_time) >= timeout_sec:
                    return False

        time.sleep(0.1)
        return True

    finally:
        observer.stop()
        observer.join()


def test_lock_hold():
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


def test_try_lock_only():
    print("Python: LockAppPref 試行")
    ok = LockAppPref()
    print("Python: Lock result =", ok)

    if ok:
        UnlockAppPref()
        print("Python: すぐUnlockしました")


def test_write_mode():
    if LockAppPref():
        try:
            ok = SetAppPref({"Mode": "2", "Ready": "1"})
            print("Python: SetAppPref result =", ok)
        finally:
            UnlockAppPref()
    else:
        print("Python: ロック失敗")


def test_read_mode():
    if LockAppPref():
        try:
            mode = GetAppPref("Mode")
            ready = GetAppPref("Ready")
            print("Python: Mode =", mode)
            print("Python: Ready =", ready)
        finally:
            UnlockAppPref()
    else:
        print("Python: ロック失敗")


def test_wait_and_read():
    print("Python: C#からの更新待ちを開始します")
    updated = WaitAppPrefUpdated(30)
    print("Python: Wait result =", updated)

    if updated:
        if LockAppPref():
            try:
                mode = GetAppPref("Mode")
                ready = GetAppPref("Ready")
                print("Python: updated Mode =", mode)
                print("Python: updated Ready =", ready)
            finally:
                UnlockAppPref()
        else:
            print("Python: 更新後ロック失敗")
    else:
        print("Python: タイムアウトしました")


if __name__ == "__main__":
    print("=== Python Test Menu ===")
    print("1: ロックして10秒保持")
    print("2: ロックできるか試すだけ")
    print("3: Mode=2, Ready=1 に書き込む")
    print("4: Mode, Ready を読む")
    print("5: C#更新待ちして読込む")
    choice = input("番号を入力してください: ").strip()

    if choice == "1":
        test_lock_hold()
    elif choice == "2":
        test_try_lock_only()
    elif choice == "3":
        test_write_mode()
    elif choice == "4":
        test_read_mode()
    elif choice == "5":
        test_wait_and_read()
    else:
        print("不正な入力です")