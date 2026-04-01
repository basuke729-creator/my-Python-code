using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Threading;

public class CameraInfo
{
    public string AppPrefPath { get; set; } = "";
}

public class AppPreferenceManager
{
    private FileStream? _lockStream = null;
    private string? _currentPath = null;

    public bool LockApplicationPreference()
    {
        try
        {
            if (_lockStream != null)
            {
                return false;
            }

            if (string.IsNullOrEmpty(_currentPath))
            {
                return false;
            }

            if (!File.Exists(_currentPath))
            {
                return false;
            }

            _lockStream = new FileStream(
                _currentPath,
                FileMode.Open,
                FileAccess.ReadWrite,
                FileShare.None
            );

            return true;
        }
        catch
        {
            if (_lockStream != null)
            {
                try
                {
                    _lockStream.Close();
                }
                catch
                {
                }

                _lockStream = null;
            }

            return false;
        }
    }

    public void UnlockApplicationPreference()
    {
        try
        {
            if (_lockStream != null)
            {
                _lockStream.Close();
            }
        }
        catch
        {
        }
        finally
        {
            _lockStream = null;
        }
    }

    public bool SetApplicationPreference(CameraInfo cameraInfo, string dataName, string type, string value)
    {
        try
        {
            if (cameraInfo == null)
            {
                return false;
            }

            _currentPath = cameraInfo.AppPrefPath;

            if (string.IsNullOrEmpty(_currentPath))
            {
                return false;
            }

            if (!LockApplicationPreference())
            {
                return false;
            }

            string jsonText;
            using (var reader = new StreamReader(_lockStream!, Encoding.UTF8, true, 1024, true))
            {
                _lockStream!.Seek(0, SeekOrigin.Begin);
                jsonText = reader.ReadToEnd();
            }

            if (string.IsNullOrWhiteSpace(jsonText))
            {
                return false;
            }

            var jsonData = JsonSerializer.Deserialize<Dictionary<string, Dictionary<string, object>>>(jsonText);

            if (jsonData == null)
            {
                return false;
            }

            if (!jsonData.ContainsKey(dataName) || jsonData[dataName] == null)
            {
                return false;
            }

            jsonData[dataName]["PrefType"] = type;
            jsonData[dataName]["Value"] = value;

            string outputJson = JsonSerializer.Serialize(jsonData, new JsonSerializerOptions
            {
                WriteIndented = true
            });

            _lockStream!.SetLength(0);
            _lockStream.Seek(0, SeekOrigin.Begin);

            using (var writer = new StreamWriter(_lockStream, new UTF8Encoding(false), 1024, true))
            {
                writer.Write(outputJson);
                writer.Flush();
            }

            _lockStream.Flush(true);

            return true;
        }
        catch
        {
            return false;
        }
        finally
        {
            UnlockApplicationPreference();
        }
    }

    public string? GetApplicationPreference(CameraInfo cameraInfo, string dataName, string type)
    {
        try
        {
            if (cameraInfo == null)
            {
                return null;
            }

            _currentPath = cameraInfo.AppPrefPath;

            if (string.IsNullOrEmpty(_currentPath))
            {
                return null;
            }

            if (!LockApplicationPreference())
            {
                return null;
            }

            string jsonText;
            using (var reader = new StreamReader(_lockStream!, Encoding.UTF8, true, 1024, true))
            {
                _lockStream!.Seek(0, SeekOrigin.Begin);
                jsonText = reader.ReadToEnd();
            }

            if (string.IsNullOrWhiteSpace(jsonText))
            {
                return null;
            }

            var jsonData = JsonSerializer.Deserialize<Dictionary<string, Dictionary<string, object>>>(jsonText);

            if (jsonData == null)
            {
                return null;
            }

            if (!jsonData.ContainsKey(dataName))
            {
                return null;
            }

            var item = jsonData[dataName];
            if (item == null)
            {
                return null;
            }

            if (item.ContainsKey("PrefType"))
            {
                string prefType = item["PrefType"]?.ToString() ?? "";
                if (!string.Equals(prefType, type, StringComparison.Ordinal))
                {
                    return null;
                }
            }

            if (!item.ContainsKey("Value"))
            {
                return null;
            }

            return item["Value"]?.ToString();
        }
        catch
        {
            return null;
        }
        finally
        {
            UnlockApplicationPreference();
        }
    }

    public bool TestLock(CameraInfo cameraInfo)
    {
        _currentPath = cameraInfo.AppPrefPath;
        return LockApplicationPreference();
    }

    public void TestUnlock()
    {
        UnlockApplicationPreference();
    }
}

class Program
{
    static void Main()
    {
        var cameraInfo = new CameraInfo
        {
            AppPrefPath = @"C:\i-pro\27_PC版対応\00_調査\jsonsample\AppPrefs.json"
        };

        var manager = new AppPreferenceManager();

        Console.WriteLine("=== C# Production JSON Test Menu ===");
        Console.WriteLine("1: ロックして10秒保持");
        Console.WriteLine("2: ロックできるか試すだけ");
        Console.WriteLine("3: Mode=2, CameraStatus=0 に更新");
        Console.WriteLine("4: Mode, CameraStatus を読む");
        Console.WriteLine("5: Mode=1, CameraStatus=1 に戻す");
        Console.Write("番号を入力してください: ");
        string? choice = Console.ReadLine();

        if (choice == "1")
        {
            Console.WriteLine("C#: Lock開始");
            bool ok = manager.TestLock(cameraInfo);
            Console.WriteLine("C#: Lock result = " + ok);

            if (ok)
            {
                try
                {
                    Console.WriteLine("C#: 10秒ロック保持します。今のうちにPythonでロックを試してください。");
                    Thread.Sleep(10000);
                }
                finally
                {
                    manager.TestUnlock();
                    Console.WriteLine("C#: Unlock 完了");
                }
            }
        }
        else if (choice == "2")
        {
            Console.WriteLine("C#: Lock試行");
            bool ok = manager.TestLock(cameraInfo);
            Console.WriteLine("C#: Lock result = " + ok);

            if (ok)
            {
                manager.TestUnlock();
                Console.WriteLine("C#: すぐUnlockしました");
            }
        }
        else if (choice == "3")
        {
            bool ok1 = manager.SetApplicationPreference(cameraInfo, "Mode", "Integer", "2");
            Console.WriteLine("C#: Set Mode result = " + ok1);

            bool ok2 = manager.SetApplicationPreference(cameraInfo, "CameraStatus", "Integer", "0");
            Console.WriteLine("C#: Set CameraStatus result = " + ok2);
        }
        else if (choice == "4")
        {
            string? mode = manager.GetApplicationPreference(cameraInfo, "Mode", "Integer");
            string? cameraStatus = manager.GetApplicationPreference(cameraInfo, "CameraStatus", "Integer");

            Console.WriteLine("C#: Mode = " + mode);
            Console.WriteLine("C#: CameraStatus = " + cameraStatus);
        }
        else if (choice == "5")
        {
            bool ok1 = manager.SetApplicationPreference(cameraInfo, "Mode", "Integer", "1");
            Console.WriteLine("C#: Restore Mode result = " + ok1);

            bool ok2 = manager.SetApplicationPreference(cameraInfo, "CameraStatus", "Integer", "1");
            Console.WriteLine("C#: Restore CameraStatus result = " + ok2);
        }
        else
        {
            Console.WriteLine("不正な入力です");
        }
    }
}