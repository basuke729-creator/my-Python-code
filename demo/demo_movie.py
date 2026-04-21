import cv2
import os
import csv
import time
import shutil
import glob
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta
import multiprocessing
import queue
from collections import deque
import onnxruntime
from motpy import Detection, MultiObjectTracker
# from bytetracker import BYTETracker
from structure import Person
import sys

SIZE = (640, 480)
ESC_KEY = 27
CAM_NO = 0
STATE_EXIT = -1
STATE_MOTION = 1

SHOW_TH = 0.1
OBJECT_CNT = 5

DETECTION_OUTSIZE_HEIGHT = 640
DETECTION_OUTSIZE_WIDTH = 640
POSE_OUTSIZE_HEIGHT = 256
POSE_OUTSIZE_WIDTH = 192
POSE_HEATMAP_HEIGHT = 64
POSE_HEATMAP_WIDTH = 48
MEAN_POSE = (0.485 * 255, 0.456 * 255, 0.406 * 255)
STD_POSE = (0.229 * 255, 0.224 * 255, 0.225 * 255)

# DETECTION_MODEL_PATH = os.path.dirname(__file__) + r"/model/yolox_nano.onnx"
# DETECTION_MODEL_PATH = os.path.dirname(__file__) + r"/model/yolox_tiny.onnx"
DETECTION_MODEL_PATH = os.path.dirname(__file__) + r"/model/yolox_s.onnx"
POSE_ESTIMATION_MODEL_PATH = os.path.dirname(__file__) + r"/model/hrnet_w32_256x192.onnx"

input_csv = "rectangles.csv"
FRAME_RATE = 5

def confplot(image, land, conf, flag):
    def cood(image, pos):
        return (int(image.shape[1] * pos[0]), int(image.shape[0] * pos[1]))

    lines = [[0, 1], [0, 2], [1, 3], [2, 4], [0, 5], [0, 6], [5, 6], [5, 7], [7, 9],
             [6, 8], [8, 10], [11, 12], [5, 11], [11, 13], [13, 15], [6, 12], [12, 14], [14, 16]]
    lands = land.reshape(-1, 2)

    
    # 危険エリア内だと赤
    if flag == True:
        color = (0, 0, 255)
    # 危険エリア外だと緑
    else:
        color = (0, 255, 0)

    for lnd in lines:
        image = cv2.line(image, cood(image, lands[lnd[0]]), cood(image, lands[lnd[1]]), color, 5)

    for p, c in zip(lands, conf):
        image = cv2.circle(image, cood(image, p), 10, color, -1)

    return image

def arrow_draw(image, bbox, color):
    def cood(image, pos):
        return (int(image.shape[1] * pos[0]), int(image.shape[0] * pos[1]))

    def clr(color, conf):
        return (int(conf * color[0]), int(conf * color[1]), int(conf * color[2]))

    area = bbox.copy()
    # bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax, bbox_score
    image = cv2.arrowedLine(image, cood(image, [area[0], area[1]]),cood(image, [area[2], area[3]]),clr(color, 1.0), 5, tipLength=0.3)

    return image

def line_draw(image, bbox, color):
    def cood(image, pos):
        return (int(image.shape[1] * pos[0]), int(image.shape[0] * pos[1]))

    def clr(color, conf):
        return (int(conf * color[0]), int(conf * color[1]), int(conf * color[2]))

    area = bbox.copy()

    image = cv2.line(image, cood(image, [area[0], area[1]]),cood(image, [area[2], area[3]]),clr(color, 1.0), 5)

    return image

def area_draw(image, bbox, color):
    def cood(image, pos):
        return (int(image.shape[1] * pos[0]), int(image.shape[0] * pos[1]))

    def clr(color, conf):
        return (int(conf * color[0]), int(conf * color[1]), int(conf * color[2]))

    area = bbox.copy()


    image = cv2.line(image, cood(image, [area[0], area[1]]), cood(image, [area[2], area[1]]), clr(color, 1.0), 5)
    image = cv2.line(image, cood(image, [area[2], area[1]]), cood(image, [area[2], area[3]]), clr(color, 1.0), 5)
    image = cv2.line(image, cood(image, [area[2], area[3]]), cood(image, [area[0], area[3]]), clr(color, 1.0), 5)
    image = cv2.line(image, cood(image, [area[0], area[3]]), cood(image, [area[0], area[1]]), clr(color, 1.0), 5)

    return image


def class_draw(image, bbox, cls):
    def cood(image, pos):
        return (int(image.shape[1] * pos[0]), int(image.shape[0] * pos[1]))

    def clr(color, conf):
        return (int(conf * color[0]), int(conf * color[1]), int(conf * color[2]))

    # 判定エリアを通った後
    if cls[3] == True:
        if cls[2] == True:
            color = (255, 0, 0)
        else:
            color = (0, 0, 255)
    else:
        if cls[2] == True:
            color = (0, 255, 0)
        else:
            color = (49, 150, 0)

    text = "id:" + str(cls[1])


    image = cv2.line(image, cood(image, [bbox[0], bbox[1]]), cood(image, [bbox[2], bbox[1]]), clr(color, cls[0]), 5)
    image = cv2.line(image, cood(image, [bbox[2], bbox[1]]), cood(image, [bbox[2], bbox[3]]), clr(color, cls[0]), 5)
    image = cv2.line(image, cood(image, [bbox[2], bbox[3]]), cood(image, [bbox[0], bbox[3]]), clr(color, cls[0]), 5)
    image = cv2.line(image, cood(image, [bbox[0], bbox[3]]), cood(image, [bbox[0], bbox[1]]), clr(color, cls[0]), 5)

    cv2.putText(image, text, cood(image, [bbox[0], bbox[1]]),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, color)

    return image


def cap_proc_fun(detect_queue, motion_queue, state_queue, result_queue):
    cap = cv2.VideoCapture(CAM_NO, cv2.CAP_MSMF)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SIZE[1])
    # size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # fps = cap.get(cv2.CAP_PROP_FPS)
    state = STATE_MOTION

    while True:
        try:
            state = state_queue.get(block=False)
        except queue.Empty:
            pass

        ret, raw_image = cap.read()
        if ret:
            if state == STATE_MOTION:
                motion_queue.put(raw_image)
            elif state == STATE_EXIT:
                break
        # rgb_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        # cv2.imshow('camera', raw_image)
        key = cv2.waitKey(1)
        if key == ESC_KEY:
            break

    detect_queue.put(None)
    motion_queue.put(None)
    result_queue.put(None)
    # cv2.destroyWindow('camera')

    return


def video_proc_fun(path, motion_queue, state_queue):
    cap = cv2.VideoCapture(path)
    state = STATE_MOTION
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = 1.0 / fps
    proc_cnts = 0
    last_time = time.time()

    # st = 23
    # ed = 40
    ret = True
    while ret:
        try:
            state = state_queue.get(block=False)
        except queue.Empty:
            pass

        now_time = time.time()
        # if proc_cnts >= 3:
        #     if (0 < ((now_time - start_time))):
        #         print(1 / (now_time - start_time))
        #     read_num = max(int((now_time - start_time) / interval), 1)
        # else:
        #     read_num = 1

        read_num = int(fps / FRAME_RATE)
        for _ in range(read_num):
            ret, raw_image = cap.read()

        # frame_no = cap.get(cv2.CAP_PROP_POS_FRAMES)
        # if(frame_no < st*fps):
        #     continue
        # elif(ed*fps < frame_no):
        #     motion_queue.put(None)
        #     break

        # # 5fpsで再生する
        # while(time.time() - last_time < 1 / (fps / read_num)):
        #     time.sleep(0.01)


        last_time = time.time()

        if ret:
            if state == STATE_MOTION:
                motion_queue.put(raw_image)
            elif state == STATE_EXIT:
                break

            # rgb_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
            # cv2.imshow('video', raw_image)
            key = cv2.waitKey(1)
            if key == ESC_KEY:
                break
        else:
            break

        start_time = time.time()
        proc_cnts += 1

    motion_queue.put(None)
    # cv2.destroyWindow('video')

    sys.stderr.write("video_proc fin\n")
    return



def overlap_ratio(rect1, rect2):
    """
    rect1, rect2 は (x_min, y_min, x_max, y_max) のタプル
    この関数は rect1 と rect2 が重なる部分の面積を rect1 の面積で割った割合を返す
    """

    # rect1 の面積
    area1 = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
    if area1 <= 0:
        return 0.0  # rect1が面積0なら0

    # 重なり領域の座標
    overlap_x_min = max(rect1[0], rect2[0])
    overlap_y_min = max(rect1[1], rect2[1])
    overlap_x_max = min(rect1[2], rect2[2])
    overlap_y_max = min(rect1[3], rect2[3])

    # 重なりの幅と高さ
    overlap_width = overlap_x_max - overlap_x_min
    overlap_height = overlap_y_max - overlap_y_min

    # 幅または高さが負なら重なりはない
    if overlap_width <= 0 or overlap_height <= 0:
        overlap_area = 0.0
    else:
        overlap_area = overlap_width * overlap_height

    # 割合を返す
    return overlap_area / area1



def orientation(p, q, r):
    """3点の向きを返す
    return: 0 -> 共線, 1 -> 時計回り, 2 -> 反時計回り
    """
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0:
        return 0  # collinear
    return 1 if val > 0 else 2  # clock or counterclockwise

def on_segment(p, q, r):
    """p,q,rが共線の時、qがprの間にあるか判定"""
    if min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1]):
        return True
    return False

def segments_intersect(p1, q1, p2, q2):
    """
    2つの線分 (p1, q1) と (p2, q2) が交差しているか判定
    """
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # 一般ケース
    if o1 != o2 and o3 != o4:
        return True

    # 特殊ケース（共線で重なっている）
    if o1 == 0 and on_segment(p1, p2, q1):
        return True
    if o2 == 0 and on_segment(p1, q2, q1):
        return True
    if o3 == 0 and on_segment(p2, p1, q2):
        return True
    if o4 == 0 and on_segment(p2, q1, q2):
        return True

    return False


def reg1dim(x, y):
    n = len(x)
    a = ((np.dot(x, y) - y.sum() * x.sum() / n) /
         ((x ** 2).sum() - x.sum() ** 2 / n))
    b = (y.sum() - a * x.sum()) / n
    return a, b



def cross_product(v1, v2):
    """
    2Dベクトルの外積を計算する
    """
    return v1[0] * v2[1] - v1[1] * v2[0]

def check_intersection_line_vector_2d(p0, p1, q0, q1):
    """
    ベクトル (p0, p1) の延長線と直線 (q0, q1) が交差するか判定する（2次元で）
    p0, p1: ベクトルが通る2点の座標 [x, y]
    q0, q1: 直線が通る2点の座標 [x, y]
    """
    # ベクトル v と w の計算
    v = np.array(p1) - np.array(p0)  # ベクトル v
    w = np.array(q1) - np.array(q0)  # ベクトル w

    # 方向ベクトルがゼロの場合を排除
    if np.linalg.norm(v) == 0 or np.linalg.norm(w) == 0:
        raise ValueError("与えられた点が同一のため、方向ベクトルがゼロになります")

    # 連立一次方程式を構成 (Ax = b)
    A = np.array([v, -w]).T  # 係数行列（2x2行列）
    b = np.array(q0) - np.array(p0)  # 定数項ベクトル

    try:
        # 連立方程式を解く
        solution = np.linalg.solve(A, b)  # x = A^-1 * b
        t, s = solution

        # 範囲チェック: s（直線の範囲内か）を判定
        if 0 <= s <= 1:  # s が直線 [q0, q1] の範囲内にあるか
            # 交点の計算
            intersection_point = np.array(p0) + t * v  # ベクトルの延長線上の交点
            return True, intersection_point
        else:
            # s が [0, 1] の範囲外（直線範囲内ではない）
            return False, None
    except np.linalg.LinAlgError:
        # ベクトルが平行（行列が特異行列の場合）
        return False, None



def crossing_line_judge(bbox, judge_line, judgment_direction):
    result = False

    bbox = np.array(bbox).copy()
    if (1 < len(bbox)):
        # st = -int(FRAME_RATE)
        # if(1 < int(FRAME_RATE) <= len(bbox)):
        #     x = np.mean(bbox[st:, ::2], axis=1)
        #     y = bbox[st:, 3]

        # else:
        #     x = np.mean(bbox[:, ::2], axis=1)
        #     y = bbox[:, 3]

        # # 1秒間の最小2乗法で傾きと切片を求める

        # a, b = reg1dim(x, y)
        # A1 = (x[-1], a*x[-1]+b)  # 現在のバウンディングボックス下辺の中心
        # A2 = (x[0], a*x[0]+b)    # 1秒前のバウンディングボックス下辺の中心

        current_x = np.mean([bbox[-1][0], bbox[-1][2]])
        current_y = np.mean([bbox[-1][3], bbox[-1][3]])
        try:
            prev_x = np.mean([bbox[-2][0], bbox[-2][2]])
            prev_y = np.mean([bbox[-2][3], bbox[-2][3]])
        except:
            prev_x = current_x
            prev_y = current_y
        L1 = (current_x, current_y)  # 現在のバウンディングボックス下辺の中心
        L2 = (prev_x, prev_y)        # 1フレームのバウンディングボックス下辺の中心


        V1 = judge_line[0:2]  # 境界線の始点
        V2 = judge_line[2:4]  # 境界線の終点

        # result, intersection = check_intersection_line_vector_2d(V1, V2, L1, L2)
        result = segments_intersect(V1, V2, L1, L2)

        # 判定エリアを超えた場合は向きを確認する
        if (result == True):
            # 移動方向のベクトル
            move_vec = np.array(L1) - np.array(L2)
            # 判定方向のベクトル
            judge_vec = np.array(judgment_direction[2:4]) - np.array(judgment_direction[0:2])
            dot = np.dot(move_vec, judge_vec)
            # 逆向きなので無効
            if (dot <= 0):
                result = False
        else:
            pass

    return result



def card_read_judge(land, card_area):
    land = np.array(land)

    serch_frame = FRAME_RATE
    frame_th = int(serch_frame * 0.4)
    # serch_frame中frame_thフレーム赤枠内に手首があればカードリーダーをかざしていると判定
    if (serch_frame <= len(land)):
        # 方手首ずつ赤枠内にあるか確認
        for i in [9, 10]:
            x = land[-serch_frame:, i, 0]
            y = land[-serch_frame:, i, 1]
            xmin, ymin, xmax, ymax = card_area

            # 各フレームで条件を満たしているか判定
            insert_cnt = np.sum((xmin <= x) & (x <= xmax) & (ymin <= y) & (y <= ymax))
            # 条件を満たしたフレームの個数が閾値以上か
            if (frame_th <= insert_cnt):
                return True

    return False


def cosine_similarity(f1, f2):
    return np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2))



def SummarizeAnalysisResult(persons, st, ed):
    # 保持している時刻を取得
    timestamps = []
    for key in persons.keys():
        timestamps.extend(persons[key].timestamp)
    timestamps = sorted(set(timestamps))

    result_file = []
    if (1 < len(timestamps)):
        # 範囲内の情報のみ抽出
        if ((timestamps[0] < st) and (ed < timestamps[-1])):
            extract_timestamps = [item for item in timestamps if st <= item <= ed]

            line_1st = []
            line_2nd = []
            line_1st.extend([""])
            line_2nd.extend(["日付"])
            for _ in range(5):
                line_1st.extend(["bbox"])
                line_2nd.extend(["xmin"])
                line_2nd.extend(["ymin"])
                line_2nd.extend(["xmax"])
                line_2nd.extend(["ymax"])
                line_2nd.extend(["score"])

            line_1st.extend([""])
            line_2nd.extend(["id"])
            for i in range(17):
                line_1st.extend([i, i, i])
                line_2nd.extend(["x", "y", "尤度"])

            line_1st.extend([""])
            line_2nd.extend(["card_read"])
            line_1st.extend([""])
            line_2nd.extend(["cross_result"])

            result_file.append(line_1st)
            result_file.append(line_2nd)

            for timestamp in extract_timestamps:
                for key in persons.keys():
                    line = []
                    try:
                        idx = persons[key].timestamp.index(timestamp)
                        line.append(timestamp.strftime('%Y%m%d%H%M%S.%f')[:-3])
                        for i in range(4):
                            line.append(persons[key].box[idx][i])
                        line.append(persons[key].score)
                        line.append(persons[key].id)
                        for (x, y), s in zip(persons[key].land_pose[idx], persons[key].conf_pose[idx]):
                            line.append(x)
                            line.append(y)
                            line.append(s)
                        line.append(int(persons[key].card_read))
                        line.append(int(persons[key].cross_result))
                        result_file.append(line)

                    except ValueError:
                        pass
    return result_file

def exec_motion_fun(motion_queue, result_queue, csv_name):
    # print(onnxruntime.get_device())
    providers = ['CUDAExecutionProvider']
    session_detection = onnxruntime.InferenceSession(DETECTION_MODEL_PATH, providers=providers)
    session_pose = onnxruntime.InferenceSession(POSE_ESTIMATION_MODEL_PATH, providers=providers)

    # 使用しているプロバイダー（この場合はCUDA）を表示
    # print(session_detection.get_providers())
    # print(session_pose.get_providers())

    # id置換用の辞書とカウンタ
    id_map = {}  # {motpy_id: friendly_id}
    next_id = 1

    persons = {}
    now = datetime.now()  # 現在時刻の取得
    st = now + timedelta(seconds=5.0)
    ed = now + timedelta(seconds=10.0)

    # カードリーダー位置などの読出し
    with open(csv_name, encoding='UTF-8') as f:
        reader = csv.reader(f)

        # for i, line in enumerate(reader):
        #     # カードリーダーの座標を取得
        #     if (i == 0):
        #         card_area = [float(v) for v in line]
        #     elif (i == 1):
        #         pass_area = [float(v) for v in line]
        #     elif (i == 2):
        #         judge_line = [float(v) for v in line]
        #     elif (i == 3):
        #         judgment_direction = [float(v) for v in line]
        #     else:
        #         pass

    # tracker_type = 1  # 0:なし, 1:motpy, 2:bytetracker

    # if (tracker_type == 1):
    #     # step_timeを秒単位で指定し、Trackerを初期化
    #     tracker = MultiObjectTracker(dt=1 / FRAME_RATE,
    #                                  tracker_kwargs={
    #                                      'max_staleness': 10,  # ロストを許容するフレーム数（デフォルトは10）
    #                                      'model_kwargs': {
    #                                          'order_pos': 2,      # 位置の変化（加速度を含む）
    #                                          'dim_pos': 2,        # 位置の次元（x, y）
    #                                          'order_size': 1,     # サイズ変化の追跡
    #                                          'dim_size': 2        # サイズの次元（width, height）
    #                                      },
    #                                  },
    #                                  matching_fn_kwargs={
    #                                      'min_iou': 0.0001,
    #                                      'feature_similarity_fn': cosine_similarity,  # 特徴量類似性計算
    #                                      'feature_similarity_beta': 0.5  # 特徴量の重要性 (0.0〜1.0)
    #                                  }
    #                                  )
    # elif (tracker_type == 2):
    #     tracker = BYTETracker(track_thresh=0.4,  # 信頼度閾値
    #                           track_buffer=30,   # トラック保持フレーム数
    #                           match_thresh=0.9,  # マッチング閾値
    #                           frame_rate=FRAME_RATE)  # フレームレート

    # else:
    #     pass


    fn = 0
    while True:
        if (fn % 100 == 0):
            print(fn)
        exit_bool = False

        try:
            image = motion_queue.get()
            if exit_bool or image is None:
                break

        except queue.Empty:
            print("timeout")
            break

        fn += 1

        origin_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        now = now + timedelta(seconds=0.2)
        print(now)

        # 物体検出の実施
        img, ratio = preprocess(origin_img, (DETECTION_OUTSIZE_HEIGHT, DETECTION_OUTSIZE_WIDTH))
        onnx_runtime_inputs = {session_detection.get_inputs()[0].name: img[None, :, :, :]}
        detected_info = session_detection.run([session_detection.get_outputs()[0].name], onnx_runtime_inputs)

        final_boxes, final_scores, final_cls_inds = get_detection_result(detected_info[0],
                                                                         (DETECTION_OUTSIZE_HEIGHT, DETECTION_OUTSIZE_WIDTH), ratio)

        if (0 < len(final_boxes)):
            bbox_ratio = np.array([1 / ratio / origin_img.shape[1], 1 / ratio / origin_img.shape[0]])
            # final_boxes = adjust_bboxes_for_occlusion(final_boxes, bbox_ratio)


        # DETECTION_TH = 0.5
        # # 物体追跡の実施
        # ids = [-1] * 6
        # start_time = time.time()  # 計測開始
        # if (tracker_type == 1):
        #     outputs = [Detection(box=box, score=score, class_id=inds) for (box, score, inds) in zip(final_boxes[DETECTION_TH < final_scores],
        #                                                                                               final_scores[DETECTION_TH < final_scores],
        #                                                                                               final_cls_inds[DETECTION_TH < final_scores])]
        #     tracker.step(detections=outputs)
        #     tracks = tracker.active_tracks(min_steps_alive=1)
        #
        #     if (0 < len(tracks)):
        #         ids = [-1] * len(tracks)
        #         for i, t in enumerate(tracks):
        #             motpy_id = t[0]
        #             if motpy_id not in id_map:
        #                 id_map[motpy_id] = next_id  # 新規IDなら割り当て
        #                 if (next_id < 1000):
        #                     next_id += 1
        #                 else:
        #                     next_id = 1
        #
        #             ids[i] = id_map[motpy_id]
        #
        #         final_boxes = np.array([t[1] for t in tracks])
        #         final_scores = np.array([t[2] for t in tracks])
        #         final_cls_inds = np.array([t[3] for t in tracks])
        #     else:
        #         ids = np.zeros(0)
        #         final_boxes = np.zeros((0, 4))
        #         final_scores = np.zeros(0)
        #         final_cls_inds = np.zeros(0)

        # elif (tracker_type == 2):
        #     dets = np.hstack([final_boxes, final_scores.reshape(-1, 1), final_cls_inds.reshape(-1, 1)])
        #     online_targets = tracker.update(dets, "-")
        #
        #     if (0 < len(online_targets)):
        #         ids = online_targets[:, 4]
        #         final_boxes = online_targets[:, :4]
        #         final_scores = online_targets[:, 6]
        #         final_cls_inds = online_targets[:, 5]
        #     else:
        #         ids = np.zeros(0)
        #         final_boxes = np.zeros((0, 4))
        #         final_scores = np.zeros(0)
        #         final_cls_inds = np.zeros(0)
        # else:
        #     final_boxes = final_boxes[DETECTION_TH < final_scores]
        #     final_cls_inds = final_cls_inds[DETECTION_TH < final_scores]
        #     final_scores = final_scores[DETECTION_TH < final_scores]
        #     ids = np.arange(len(final_boxes))

        # end_time = time.time()  # 計測終了
        # elapsed_time_ms = (end_time - start_time) * 1000
        # print(f"処理時間: {elapsed_time_ms:.3f}ミリ秒")


        # 枠外の場合は枠内におさめる
        final_boxes[final_boxes < 0.0] = 0.0

        # xyxy2xywh
        final_boxes_converted = convert_bboxes_info(final_boxes.copy())
        # バウンディングボックスの中心と幅を算出
        centers, scales = calculate_center_and_scale(final_boxes_converted)

        # Normalize for pose estimation
        img[0] = (img[0] - MEAN_POSE[0]) / STD_POSE[0]
        img[1] = (img[1] - MEAN_POSE[1]) / STD_POSE[1]
        img[2] = (img[2] - MEAN_POSE[2]) / STD_POSE[2]

        # バウンディングボックスの座標を元画像基準に変換
        bbox_ratio = np.array([1 / ratio / origin_img.shape[1], 1 / ratio / origin_img.shape[0]])
        joints = np.zeros((len(final_boxes), 17, 2))
        scores_list = np.zeros((len(final_boxes), 17))
        # BBoxごとにカードリーダー近ければ姿勢推定、判定ラインを超えたか判定の実施
        for i in range(len(final_boxes)):
            # オリジナル画像サイズに変換
            # final_box = final_boxes[i].copy() * np.tile(bbox_ratio, 2)

            # 通過エリアに対するバウンディングボックスとの重複割合を算出
            # overlap_rate = overlap_ratio(final_box, pass_area)
            # print(overlap_rate)
            overlap_rate = 1.0
            try:
                # バウンディングボックスが判定範囲内に来た場合
                if (0.6 < overlap_rate):
                    img_in_bbox = cv2.resize(
                        img[:, int(final_boxes[i, 1]):int(final_boxes[i, 3] + 1), int(final_boxes[i, 0]):int(final_boxes[i, 2] + 1)].transpose(1, 2, 0),
                        (POSE_OUTSIZE_WIDTH, POSE_OUTSIZE_HEIGHT)
                    )
                    # hwcからchwに変換
                    img_in_bbox = img_in_bbox.transpose(2, 0, 1)

                    onnx_runtime_inputs = {session_pose.get_inputs()[0].name: img_in_bbox[None, :, :, :]}
                    heatmaps = session_pose.run([session_pose.get_outputs()[0].name], onnx_runtime_inputs)[0]

                    preds, score = get_pose_estimation_result(heatmaps, centers[i], scales[i], ratio)

                    joints[i, :, :] = preds
                    scores_list[i, :] = score
            except:
                pass


        # エリアの描画(area_setting.pyのcv2.polylinesを参考に変更)
        # area_draw(image, card_area, (0, 255, 0))

        # 登録している人がロスされていなければ、判定と描画を行う
        for land_pose, conf_pose in zip(joints, scores_list):
            # エリア内判定
            flag = True

            # 骨格描画(land poseは0~1にしてください)
            confplot(image, land_pose, conf_pose, flag)

        result_queue.put(image)


    result_queue.put(None)
    # time.sleep(3.0)
    sys.stderr.write("motion_proc fin\n")
    return

def iou(boxA, boxB):
    """IoU計算: 正規化座標想定 (x_min, y_min, x_max, y_max)"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def correct_occluded_bbox(front_bbox, occluded_bbox, bbox_ratio,
                          height_ratio=0.8, threshold_min=0.8, eps=1e-6):
    """
    手前と奥のBBox関係で奥側の高さを補正
    """
    front_height = (front_bbox[3] - front_bbox[1]) * height_ratio
    occ_height = occluded_bbox[3] - occluded_bbox[1]

    # if (front_bbox[3]*bbox_ratio[1] >= 0.95):
    #     # 手前が端まで到達 → 奥も端まで
    #     corrected_bbox = (occluded_bbox[0], occluded_bbox[1],
    #                       occluded_bbox[2], front_bbox[3])
    # else:
    # 奥側の高さが手前の8割しかない場合、手前側と同じ高さにする
    if (occ_height + eps < front_height * threshold_min):
        x_min, y_min, x_max, _ = occluded_bbox
        y_max = min(y_min + front_height * (1 / bbox_ratio[1]), 1)
        corrected_bbox = (x_min, y_min, x_max, y_max)
    else:
        corrected_bbox = occluded_bbox

    return corrected_bbox


def adjust_bboxes_for_occlusion(boxes, bbox_ratio, iou_threshold=0.1):
    """
    tracked_objects: [(id, bbox), ...]  bboxは0~1正規化
    iou_threshold: オクルージョン判定用IoUしきい値
    """
    corrected = {}

    for i, box_a in enumerate(boxes):
        for j, box_b in enumerate(boxes):
            if i == j:
                continue
            # 重なり判定
            if iou(box_a, box_b) > iou_threshold:
                # 手前・奥判定（y_minが大きい方が手前）
                if box_a[1] > box_b[1]:
                    front, occluded = box_a, box_b
                    occluded_id = j
                else:
                    front, occluded = box_b, box_a
                    occluded_id = i

                corrected[occluded_id] = correct_occluded_bbox(front, occluded, bbox_ratio)

    # 最終的な結果: 補正があれば上書き
    results = []
    for obj_id, bbox in enumerate(boxes):
        if obj_id in corrected:
            results.append(corrected[obj_id])
        else:
            results.append(bbox)
    return np.array(results)



def update_person(now, keypoints_list, scores_list, bbox_list, ids, persons, tracker_type):



    # diff = []


    # # 登録済みの人と新規のバウンディングボックスの差分を求める
    # for i, key in enumerate(persons.keys()):
    #     src_box = persons[key].box


    #     s = src_land[-1].reshape(-1, 2)
    #     pos_avg_s = np.average(s[COCO17_HUMAN_TRUNK], axis=0)
    #     for j, (dst_land, dst_box) in enumerate(zip(keypoints_list, bbox_list)):
    #         d = np.array(dst_land)
    #         pos_avg_d = np.average(d[COCO17_HUMAN_TRUNK], axis=0)
    #         diff.append((np.linalg.norm(pos_avg_s - pos_avg_d) + np.linalg.norm(np.array(src_box[-1]) - np.array(dst_box)), (i, j)))


    # diff = sorted(diff, key=lambda x: x[0])
    # s_in = [True] * len(persons)
    # d_in = [True] * len(keypoints_list)
    # match_list = []
    # for d in diff:
    #     if s_in[d[1][0]] and d_in[d[1][1]]:
    #         match_list.append(d[1])
    #         s_in[d[1][0]] = False
    #         d_in[d[1][1]] = False


    # match_list
    # # for m in match_list:
    # #     i = m[0]
    # #     j = m[1]
    # #     all_land_poses[i].append(np.array(keypoints_list[j]).reshape(-1))
    # #     all_conf_poses[i].append(scores_list[j])
    # #     all_bboxes[i].append(bbox_list[j])


    # else:
    # 追跡情報の更新
    for i in range(len(ids)):
        if (ids[i] != -1):
            # 登録済みの追跡対象が見つかった場合
            if (ids[i] in persons.keys()):
                box = bbox_list[i][:4]
                # 骨格検出を行っており足首の尤度が高い場合はスムージング無し
                # if ((0 < len(scores_list[i])) and ((0.3 < scores_list[i][15]) or (0.3 < scores_list[i][16]))):
                #     pass
                # else:
                #     # w, hのスムージング
                #     if (0.05 < box[0]) and (0.05 < box[1]) and (box[2] < 0.95) and (box[3] < 0.95):
                #         before_w = persons[ids[i]].box[-1][2] - persons[ids[i]].box[-1][0]
                #         before_h = persons[ids[i]].box[-1][3] - persons[ids[i]].box[-1][1]
                #         w = bbox_list[i][2] - bbox_list[i][0]
                #         h = bbox_list[i][3] - bbox_list[i][1]
                #
                #         print(before_h)
                #         a = 1 / (FRAME_RATE * 5)
                #         box[2] = box[0] + (a * w + (1 - a) * before_w)
                #         box[3] = box[1] + (a * h + (1 - a) * before_h)

                # 他の人とiouを求め、重なっているようであれば、hは面積の大きい方を採用
                # for j in range(len(bbox_list)):
                #     if(i != j):
                



                persons[ids[i]].timestamp.append(now)

                # box_heights = [box[3] - box[1] for box in persons[ids[i]].box]
                # for j, box_height in enumerate(box_heights):
                #     if(j == 0):
                #         correct_height = box_height
                #     else:
                #         correct_height = correct_height * 0.9 + box_height * 0.1
                # box[3] = box[1] + correct_height

                # box_prev = persons[ids[i]].box[-1]
                # height_prev = box_prev[3] - box_prev[1]
                # height = box[3] - box[1]
                # correct_height = height_prev * 0.95 + height * 0.05
                # box[3] = box[1] + correct_height

                persons[ids[i]].box.append(box)
                persons[ids[i]].score = bbox_list[i][4]
                persons[ids[i]].id = ids[i]
                persons[ids[i]].land_pose.append(keypoints_list[i])
                persons[ids[i]].conf_pose.append(scores_list[i])
            # 未登録の追跡対象が見つかった場合
            else:
                persons[ids[i]] = Person(now, bbox_list[i][:4], bbox_list[i][4], keypoints_list[i], scores_list[i], ids[i])

    # オクルージョンの補正
    # if(0 < len(persons)):
    #     adjusted = adjust_bboxes_for_occlusion([(persons[key].id, persons[key].box[-1]) for key in persons.keys()])
    #     for id, box in adjusted:
    #         persons[id].box[-1] = box


    # 追跡中の人をロストしているフレーム数をカウント
    for key in persons.copy():
        if (key in ids):
            persons[key].lost_cnt = 0
        else:
            persons[key].lost_cnt += 1

            # 検出できなかった人は同じ位置にいるとする
            persons[key].box.append(persons[key].box[-1])
            persons[key].land_pose.append(persons[key].land_pose[-1])
            persons[key].conf_pose.append(persons[key].conf_pose[-1])

            # 5秒間ロストした人は削除
            if (5 * FRAME_RATE < persons[key].lost_cnt):
                del persons[key]

    return persons



def imwrite(filename, img, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)

        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False

def exec_result_fun(result_queue, state_queue, basename, movie_name):
    writer = None
    filename = os.path.splitext(os.path.basename(movie_name))[0]

    if (not os.path.isdir("output")):
        os.mkdir("output")

    # if(os.path.isdir("output"+basename+'/'+filename)):
    #     shutil.rmtree("output"+basename+'/'+filename)
    os.makedirs("output" + basename + '/' + filename, exist_ok=True)


    image_cnt = 0
    while True:
        image = result_queue.get()
        if image is None:
            break
        cv2.imshow('result', image)

        if writer is None:
            writer = cv2.VideoWriter("output" + basename + '/' + filename + '.mp4',
                                     cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 5,(image.shape[1], image.shape[0]))
        writer.write(image)
        imwrite(os.path.join("output" + basename + '/' + filename, str(image_cnt) + '.jpg'), image)
        image_cnt += 1
        key = cv2.waitKey(1)
        if key == ESC_KEY:
            state_queue.put(STATE_EXIT)
            break


    cv2.destroyWindow('result')
    writer.release()

    state_queue.close()
    state_queue.join_thread()
    result_queue.close()
    result_queue.join_thread()
    # time.sleep(5.0)
    sys.stderr.write("result_proc fin\n")
    return



def demo_camera():
    detect_queue = multiprocessing.Queue()
    motion_queue = multiprocessing.Queue()
    state_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()
    cap_proc = multiprocessing.Process(target=cap_proc_fun, args=(detect_queue, motion_queue, state_queue, result_queue))
    motion_proc = multiprocessing.Process(target=exec_motion_fun, args=(motion_queue, state_queue, result_queue))
    result_proc = multiprocessing.Process(target=exec_result_fun, args=(result_queue, state_queue))
    cap_proc.start()
    motion_proc.start()
    result_proc.start()
    cap_proc.join()
    motion_proc.join()
    result_proc.join()

    return


def demo_video(dir_name, csv_name, movie_name):

    detect_queue = multiprocessing.Queue()
    motion_queue = multiprocessing.Queue()
    state_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()

    video_proc = multiprocessing.Process(target=video_proc_fun, args=(movie_name, motion_queue, state_queue), daemon=True)
    motion_proc = multiprocessing.Process(target=exec_motion_fun, args=(motion_queue, result_queue, csv_name), daemon=True)
    result_proc = multiprocessing.Process(target=exec_result_fun, args=(result_queue, state_queue, dir_name, movie_name), daemon=True)
    video_proc.start()
    motion_proc.start()
    result_proc.start()

    video_proc.join()
    print("video_proc fin")
    motion_proc.join()
    print("motion_proc fin")
    result_proc.join()
    print("result_proc fin")




def preprocess(img, input_size, swap=(2, 0, 1)):
    # img (H, W. C)
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


def get_detection_result(inference_output, input_shape, ratio):

    predictions = decode_detection_info(inference_output, input_shape)[0]

    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]

    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2. + 1
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2. + 1
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2. - 1
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2. - 1

    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.55, score_thr=0.2, class_agnostic=False)

    if dets is not None:
        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        # 人クラス(0番)のみを抽出
        # BBoxの出力は、左上のx座標、左上のy座標、高さ、幅
        final_boxes = final_boxes[final_cls_inds == 0]
        final_scores = final_scores[final_cls_inds == 0]
        final_cls_inds = final_cls_inds[final_cls_inds == 0]

        return final_boxes, final_scores, final_cls_inds
    else:
        return np.empty((0, 4)), np.empty(0), np.empty(0)


def decode_detection_info(outputs, img_size, p6=False):
    """
    Get bbox coordinates from the regression offset inferred by the model.
    """
    grids = []
    expanded_strides = []
    strides = [8, 16, 32] if not p6 else [8, 16, 32, 64]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

    return outputs


def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


def multiclass_nms(boxes, scores, nms_thr, score_thr, class_agnostic=True):
    """Multiclass NMS implemented in Numpy"""
    if class_agnostic:
        nms_method = multiclass_nms_class_agnostic
    else:
        nms_method = multiclass_nms_class_aware
    return nms_method(boxes, scores, nms_thr, score_thr)


def multiclass_nms_class_aware(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-aware version."""
    final_dets = []
    # num_classes = scores.shape[1]
    # for cls_ind in range(num_classes):
    for cls_ind in range(1):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)


def multiclass_nms_class_agnostic(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-agnostic version."""
    cls_inds = scores.argmax(1)
    cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

    valid_score_mask = cls_scores > score_thr
    if valid_score_mask.sum() == 0:
        return None
    valid_scores = cls_scores[valid_score_mask]
    valid_boxes = boxes[valid_score_mask]
    valid_cls_inds = cls_inds[valid_score_mask]
    keep = nms(valid_boxes, valid_scores, nms_thr)
    if keep:
        dets = np.concatenate(
            [valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1
        )
    return dets


def convert_bboxes_info(bboxes):
    """
    change bboxes information from (x1, y2, x2, y3) to (x, y, w, h)
    """
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0] + 1
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1] + 1
    return bboxes


def calculate_center_and_scale(bboxes):
    num_of_bboxes = len(bboxes)

    xs = bboxes[:, 0]
    ys = bboxes[:, 1]
    ws = bboxes[:, 2]
    hs = bboxes[:, 3]

    centers = np.zeros((num_of_bboxes, 2))
    centers[:, 0] = xs + ws / 2.0 - 0.5
    centers[:, 1] = ys + hs / 2.0 - 0.5

    scales = np.zeros((num_of_bboxes, 2))
    scales[:, 0] = ws
    scales[:, 1] = hs

    return centers, scales


# 高速化version
def get_pose_estimation_result(heatmaps, centers, scales, ratio):
    num_of_keypoint = 17

    max_heatmap_size = np.concatenate(
        [np.tile(POSE_HEATMAP_WIDTH, (num_of_keypoint, 1)), np.tile(POSE_HEATMAP_HEIGHT, (num_of_keypoint, 1))],
        axis=1
    )

    heatmaps_reshaped = heatmaps.reshape((num_of_keypoint, -1))
    max1_idxs = np.argpartition(heatmaps_reshaped, axis=1, kth=-1)[:, -1].reshape(-1, 1)
    # max2_idxs = np.argpartition(heatmaps_reshaped, axis=1, kth=-2)[:, -2].reshape(-1, 1)
    score = heatmaps_reshaped[np.arange(0, num_of_keypoint), max1_idxs[:, 0]].reshape(-1, 1).squeeze()

    kps = np.tile(max1_idxs, (1, 2))
    kps[:, 0] = kps[:, 0] % POSE_HEATMAP_WIDTH
    kps[:, 1] = np.floor(kps[:, 1] / POSE_HEATMAP_WIDTH)

    # kps2 = np.tile(max2_idxs, (1, 2))
    # kps2[:, 0] = kps2[:, 0] % POSE_HEATMAP_WIDTH
    # kps2[:, 1] = np.floor(kps2[:, 1] / POSE_HEATMAP_WIDTH)
    #
    # kps2_from_1 = kps2 - kps
    # ln = ((kps2_from_1[:, 0]**2 + kps2_from_1[:, 1]**2) ** 0.5).reshape((num_of_keypoint, 1))
    # kps = np.where(ln > 1e-3, kps + 0.25*kps2_from_1/ln, kps)

    kps = np.where(kps < 0, 0, np.minimum(kps, max_heatmap_size - 1, dtype=np.float32))

    # # Filtering by score
    # pred_mask = np.tile(np.where(score > threshold, 1, np.nan), (1, 2))
    # kps *= pred_mask

    # 姿勢推定モデルの入力画像上のx座標、y座標へ変換
    kps = kps * 4 + 1.5
    # 物体検出モデルの入力画像上のx座標、y座標へ変換
    ratio_x = POSE_OUTSIZE_WIDTH / scales[0]
    ratio_y = POSE_OUTSIZE_HEIGHT / scales[1]
    kps[:, 0] = kps[:, 0] / ratio_x + centers[0] - scales[0] * 0.5 + 0.5 / ratio_x
    kps[:, 1] = kps[:, 1] / ratio_y + centers[1] - scales[1] * 0.5 + 0.5 / ratio_y

    preds = kps / ratio + (1 / ratio - 1) / 2

    return preds, score


######################################################################
# execute
######################################################################
if __name__ == '__main__':
    # demo_camera()
    # demo_video(os.path.dirname(__file__) + '/videoData/WIN_20241107_13_34_44_Pro.mp4')
    # demo_video(os.path.dirname(__file__) + '/videoData/WIN_20241107_13_35_17_Pro.mp4')

    base = "C:\\Users\\yutakai\\Desktop\\demo\\input"
    target = os.path.join(base, 'videoData')

    if (os.path.isdir(target)):
        for p in Path(target).rglob('*'):
            if p.is_dir():
                dir_name = str(p)
                csv_name = glob.glob(str(p) + '/**.csv', recursive=True)
                if (csv_name != []):
                    movie_names = glob.glob(str(p) + '/**.mp4', recursive=True)
                    # print(movie_names)
                    for movie_name in movie_names:
                        print(dir_name.replace(str(target), "").replace("\\", "/"))
                        print(csv_name[0])
                        print(movie_name)
                        demo_video(dir_name.replace(str(base), "").replace("\\", "/"), csv_name[0], movie_name)
                        print("demo video fin")

    elif (os.path.isfile(target)):

        dir_name = os.path.dirname(target)
        csv_name = dir_name + '\\rectangles.csv'
        movie_name = target
        demo_video(dir_name.replace(str(base), ""), csv_name, movie_name)
        print("demo video fin")



    # demo_video(os.path.dirname(__file__) + '/videoData/01_yamamoto_bottom_normal-cam1.mp4')
    # demo_video(os.path.dirname(__file__) + '/videoData/20260107_9F_angle3_onoda.mp4')
    # demo_video(os.path.dirname(__file__) + '/walk_f/sf_miwa.mp4')
    # demo_video(os.path.dirname(__file__) + '/stay_f/sf_miwa.mp4')
