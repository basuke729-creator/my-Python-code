import cv2
import numpy as np

img_path = "skin_ref.png"  # ←参考画像のパスに変更
img = cv2.imread(img_path)
if img is None:
    raise RuntimeError(f"画像が読めません: {img_path}")

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# ノイズ対策：クリック点の周辺(11x11)平均を取る
R = 5  # 半径（5なら 11x11）
def mean_hsv(x, y):
    h, w = hsv.shape[:2]
    x0, x1 = max(0, x-R), min(w, x+R+1)
    y0, y1 = max(0, y-R), min(h, y+R+1)
    patch = hsv[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)
    return patch.mean(axis=0)

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        b, g, r = img[y, x].tolist()
        mh, ms, mv = mean_hsv(x, y)
        print(f"clicked: (x={x}, y={y})")
        print(f"  BGR(point) = {(b,g,r)}")
        print(f"  HSV(mean {2*R+1}x{2*R+1}) = (H={mh:.1f}, S={ms:.1f}, V={mv:.1f})")

cv2.namedWindow("ref", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("ref", on_mouse)

while True:
    cv2.imshow("ref", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()