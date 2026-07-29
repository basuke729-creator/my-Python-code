# Copyright (c) 2025 FUJI SOFT Inc. All rights reserved.

import os
import json
import tkinter as tk
from tkinter import messagebox
import threading
import time
import av
# av.logging.set_level(av.logging.VERBOSE)
import sys
import cv2
import numpy as np
from PIL import ImageTk, Image, ImageOps
import queue
from collections import deque
import platform
from fractions import Fraction
if platform.system() == 'Linux':
    import gi
    gi.require_version('Gst', '1.0')
    gi.require_version('GstApp', '1.0')
    from gi.repository import Gst, GstApp, GLib
    Gst.init(None)

from infer_yolox_tensorrt import main


BLANK_DUMMY_SIZE = (1920, 3840, 3)


class GUI:

    def set_param(self, setting_file):
        self.defult_param = {
            "pre_event_sec": 15,
            "post_event_sec": 15,
            "recording_path": "~/unsafe_act_monitoring/output/",
            "video_filename_format": "REC_%Y%m%d%H%M%S.mp4"
        }

        try:
            with open(setting_file, "r", encoding='utf-8') as f:
                self.param = json.load(f)
        except FileNotFoundError:
            with open("setting.json", "w", encoding='utf-8') as f:
                json.dump(self.defult_param, f, indent=4)
            self.param = self.defult_param.copy()
        except Exception as e:
            print(e)
            self.param = self.defult_param.copy()

        return

    def make_rec_container(self):
        try:
            rec_path = os.path.expanduser(self.param['recording_path'])
            os.makedirs(rec_path, exist_ok=True)
        except Exception as e:
            print(e)
            rec_path = os.path.expanduser(self.defult_param['recording_path'])
            try:
                os.makedirs(rec_path, exist_ok=True)
            except Exception as e:
                print(e)
                rec_path = ''

        try:
            rec_name = time.strftime(self.param['video_filename_format'], time.localtime())
            filename = os.path.join(rec_path, rec_name)
            out_container = av.open(filename, mode="w")
        except Exception as e:
            print(e)
            rec_name = time.strftime(self.defult_param['video_filename_format'], time.localtime())
            filename = os.path.join(rec_path, rec_name)
            out_container = av.open(filename, mode="w")

        return out_container

    def __init__(self, path=None):
        self.set_param("setting.json")

        # ---Root settings----------
        self.root = tk.Tk()
        self.root.title("unsafe act monitoring")
        self.root.resizable(False, False)
        self.root.geometry("1200x600")

        rec_test_button = tk.Button(
            self.root,
            text="Rec Test",
            width=10,
            height=1,
            padx=0,
            pady=0,
            borderwidth=3,
            relief="raised"
        )
        rec_test_button.bind("<ButtonPress>", self.rec_test)
        rec_test_button.pack(side=tk.TOP)

        self.capture_image_queue = queue.Queue()
        self.output_image_queue = queue.Queue()
        self.stream_queue = queue.Queue()
        self.latest_stream_queue = queue.Queue()
        self.latest_image_queue = queue.Queue()
        self.stream_buffer = deque()
        self.stream_a_buffer = deque()
        self.lock = threading.Lock()
        self.rec_start_event = threading.Event()
        self.rec_stop_event = threading.Event()
        self.buffer_duration_sec = 10
        self.video_stream_info = None
        self.audio_stream_info = None
        self.rec = False

        self.img_canvas_make()

        self.capture_start(path)

        self.buffer_ctrl_thread = threading.Thread(target=self.buffer_ctrl_func, args=(path,), daemon=True)
        self.buffer_ctrl_thread.start()

        self.inference_thread = threading.Thread(target=self.inference_func, daemon=True)
        self.inference_thread.start()

        self.image_update_func()

        # ---Main loop----------
        self.root.update()
        self.root.mainloop()

        print("end root")

        return

    def packet_sample(self, sink):
        sample = sink.emit("pull-sample")
        if not sample:
            return Gst.FlowReturn.OK

        buf = sample.get_buffer()
        success, map_info = buf.map(Gst.MapFlags.READ)
        if success:
            try:
                packets = self.parser_context.parse(map_info.data)
                for packet in packets:
                    self.stream_queue.put(packet)
            except Exception as e:
                print(e)
            buf.unmap(map_info)

        return Gst.FlowReturn.OK

    def video_sample(self, sink):
        sample = sink.emit("pull-sample")
        if not sample:
            return Gst.FlowReturn.OK

        caps = sample.get_caps()
        structure = caps.get_structure(0)
        width = structure.get_value("width")
        height = structure.get_value("height")

        buf = sample.get_buffer()
        success, map_info = buf.map(Gst.MapFlags.READ)
        if success:
            try:
                img_rgba = np.frombuffer(map_info.data, dtype=np.uint8).reshape((height, width, 4))
                self.capture_image_queue.put(cv2.cvtColor(img_rgba, cv2.COLOR_BGRA2BGR))
            except Exception as e:
                print(e)
            finally:
                buf.unmap(map_info)

        return Gst.FlowReturn.OK

    def gst_capture_func(self):
        pipeline_str = (
            "thetauvsrc mode=4K ! h264parse ! "
            "tee name=t "
            "t. ! queue name=q_packet max-size-buffers=10 ! "
            "video/x-h264, stream-format=byte-stream, alignment=au ! "
            "appsink name=sink_packet emit-signals=true max-buffers=10 drop=true "
            "t. ! queue name=q_video max-size-buffers=2 ! "
            "nvv4l2decoder ! nvvidconv ! video/x-raw, format=BGRx ! "
            "appsink name=sink_video emit-signals=true max-buffers=1 drop=true"
        )

        self.pipeline = Gst.parse_launch(pipeline_str)

        self.sink_packet = self.pipeline.get_by_name("sink_packet")
        self.sink_video = self.pipeline.get_by_name("sink_video")

        self.sink_packet.connect("new-sample", self.packet_sample)
        self.sink_video.connect("new-sample", self.video_sample)

        self.parser_context = av.CodecContext.create('h264', 'r')

        self.lock = threading.Lock()
        self.loop = GLib.MainLoop()

        self.pipeline.set_state(Gst.State.PLAYING)

        try:
            self.loop.run()
        except Exception as e:
            print(e)
        finally:
            self.pipeline.set_state(Gst.State.NULL)

        return

    def camera_capture_func_(self):
        try:
            if 0:  # RICOH THETA Z1
                container = av.open(format='dshow', file='video=RICOH THETA UVC:audio=マイク (RICOH THETA Z1)',
                                    options=dict(video_size='3840x1920', vcodec='h264', rtbufsize='100M'))
            else:  # test web cam
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
                cap.set(cv2.CAP_PROP_EXPOSURE, -5)
                cap.release()
                container = av.open(format='dshow',
                                    file='video=C922 Pro Stream Webcam:audio=マイク (C922 Pro Stream Webcam)',
                                    options=dict(video_size='1920x1080', vcodec='mjpeg',
                                                 framerate='30', rtbufsize='100000000'))

            dummy_container = None

            while True:
                # frames = container.decode(container.streams.video[0])
                frames = container.decode()
                for frame in frames:
                    if frame.dts is None or frame.pts is None:
                        continue

                    if dummy_container is None:
                        dummy_container = av.open("dummy", mode="w", format="null")
                        try:
                            v_stream = dummy_container.add_stream("h264_nvenc", rate=30)
                            v_stream.options = {"preset": "p3", "tune": "ll"}
                        except Exception as e:
                            print("NVENC was disabled: ", e)
                            print("Using a Software Encoder.")
                            v_stream = dummy_container.add_stream("libx264", rate=30)
                            v_stream.options = {"preset": "fast", "tune": "zerolatency"}
                        v_stream.width = container.streams.video[0].width
                        v_stream.height = container.streams.video[0].height
                        v_stream.pix_fmt = "yuv420p"
                        v_stream.bit_rate = 56000000
                        v_stream.gop_size = 30
                        v_stream.options = {
                            'forced-idr': '1',
                            'strict_gop': '1',
                            'no-scenecut': '1',
                            'b_ref_mode': '0'
                        }
                        a_stream = dummy_container.add_stream("aac", rate=48000)
                        a_stream.layout = "mono"
                        a_stream.format = "fltp"
                        a_stream.options = {"b": "96"}
                        a_stream.time_base = Fraction(1, 48000)
                        resampler = av.AudioResampler(
                            format=a_stream.format,
                            layout=a_stream.layout,
                            rate=a_stream.rate,
                        )
                        self.video_stream_info = v_stream
                        self.audio_stream_info = a_stream

                    if isinstance(frame, av.VideoFrame):
                        frame_yuv420p = frame.reformat(format='yuv420p')
                        frame_yuv420p.dts = None
                        frame_yuv420p.pts = None
                        frame_yuv420p.time_base = Fraction(1, 30)
                        for packet in v_stream.encode(frame_yuv420p):
                            dummy_container.mux(packet)
                            self.stream_queue.put(packet)
                        self.capture_image_queue.put(frame.to_ndarray(format='bgr24'))
                    elif isinstance(frame, av.AudioFrame):
                        continue
                        r_frames = resampler.resample(frame)
                        for r_frame in r_frames:
                            r_frame.dts = None
                            r_frame.pts = None
                            for packet in a_stream.encode(r_frame):
                                dummy_container.mux(packet)
                                self.stream_queue.put(packet)
        except Exception as e:
            print(e)
        finally:
            container.close()
            dummy_container.close()

        return

    def camera_capture_func(self):
        try:
            if 1:  # RICOH THETA Z1
                container = av.open(format='dshow',
                                    file='video=RICOH THETA UVC;#audio=マイク (RICOH THETA Z1)',
                                    options=dict(video_size='3840x1920', vcodec='h264', rtbufsize='100M'))
            else:  # test web cam
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
                cap.set(cv2.CAP_PROP_EXPOSURE, -5)
                cap.release()
                container = av.open(format='dshow',
                                    file='video=C922 Pro Stream Webcam;#audio=マイク (C922 Pro Stream Webcam)',
                                    options=dict(video_size='1920x1080', vcodec='mjpeg',
                                                 framerate='30', rtbufsize='100000000'))

            in_v_stream = container.streams.video[0] if container.streams.video else None
            in_a_stream = container.streams.audio[0] if container.streams.audio else None

            dummy_container = av.open("dummy", mode="w", format="null")
            try:
                v_stream = dummy_container.add_stream("h264_nvenc", rate=30)
                v_stream.options = {"preset": "p3", "tune": "ll"}
            except Exception as e:
                print("NVENC was disabled: ", e)
                print("Using a Software Encoder.")
                v_stream = dummy_container.add_stream("libx264", rate=30)
                v_stream.options = {"preset": "fast", "tune": "zerolatency"}
            v_stream.width = in_v_stream.width
            v_stream.height = in_v_stream.height
            v_stream.pix_fmt = "yuv420p"
            v_stream.bit_rate = 56000000
            v_stream.gop_size = 30
            v_stream.options = {
                'forced-idr': '1',
                'strict_gop': '1',
                'no-scenecut': '1',
                'b_ref_mode': '0'
            }
            a_stream = dummy_container.add_stream("aac", rate=48000)  # in_a_stream.rate
            a_stream.format = "fltp"
            a_stream.layout = "mono"
            a_stream.options = {"b": "96"}
            a_stream.time_base = Fraction(1, 48000)
            resampler = av.AudioResampler(
                format=a_stream.format,
                layout=a_stream.layout,
                rate=a_stream.rate,
            )
            fifo = av.AudioFifo()
            output_frame_size = 1024  # a_stream.codec_context.frame_size
            self.video_stream_info = v_stream
            self.audio_stream_info = a_stream

            while True:
                # frames = container.decode(container.streams.video[0])
                frames = container.decode()
                for frame in frames:
                    if frame.dts is None or frame.pts is None:
                        continue

                    if isinstance(frame, av.VideoFrame):
                        frame_yuv420p = frame.reformat(format='yuv420p')
                        frame_yuv420p.dts = None
                        frame_yuv420p.pts = None
                        frame_yuv420p.time_base = Fraction(1, 30)
                        for packet in v_stream.encode(frame_yuv420p):
                            dummy_container.mux(packet)
                            self.stream_queue.put(packet)
                        self.capture_image_queue.put(frame.to_ndarray(format='bgr24'))
                    elif isinstance(frame, av.AudioFrame):
                        # continue
                        r_frames = resampler.resample(frame)
                        for r_frame in r_frames:
                            r_frame.dts = None
                            r_frame.pts = None
                            fifo.write(r_frame)
                        while fifo.samples >= output_frame_size:
                            fifo_frame = fifo.read(output_frame_size)  #madadame
                            fifo_frame.dts = None
                            fifo_frame.pts = None
                            for packet in a_stream.encode(fifo_frame):
                                dummy_container.mux(packet)
                                self.stream_queue.put(packet)
        except Exception as e:
            print(e)
        finally:
            container.close()
            dummy_container.close()

        return

    def video_capture_func(self, path):
        try:
            container = av.open(path)
            v_stream = container.streams.video[0] if container.streams.video else None
            a_stream = container.streams.audio[0] if container.streams.audio else None
            self.video_stream_info = v_stream
            self.audio_stream_info = a_stream
            video_pts_offset = 0
            audio_pts_offset = 0

            while True:
                start_time = None
                for packet in container.demux():
                    if packet.dts is None or packet.pts is None:
                        continue
                    if start_time is None and v_stream and packet.stream == v_stream and packet.pts is not None:
                        start_time = time.perf_counter() - (packet.pts * float(v_stream.time_base))

                    if v_stream and packet.stream == v_stream and v_stream.duration and packet.pts < v_stream.duration:
                        packet_time = packet.pts * float(v_stream.time_base)
                        elapsed_time = time.perf_counter() - start_time
                        sleep_time = packet_time - elapsed_time
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                        packet.pts += video_pts_offset
                        packet.dts += video_pts_offset
                        self.stream_queue.put(packet)

                        frames = packet.decode()
                        for frame in frames:
                            self.capture_image_queue.put(frame.to_ndarray(format='bgr24'))
                    elif a_stream and packet.stream == a_stream and a_stream.duration and packet.pts < a_stream.duration:
                        packet.pts += audio_pts_offset
                        packet.dts += audio_pts_offset
                        self.stream_queue.put(packet)

                if v_stream:
                    video_pts_offset += v_stream.duration
                if a_stream:
                    audio_pts_offset += a_stream.duration
                container.seek(0)
                v_stream.codec_context.flush_buffers()
                a_stream.codec_context.flush_buffers()
        except Exception as e:
            print(e)
        finally:
            container.close()

        return

    def capture_start(self, path):
        if path is None:
            if platform.system() == 'Linux':
                self.codec_context = av.Codec('h264', 'r').create()
                self.capture_thread = threading.Thread(target=self.gst_capture_func, daemon=True)
            elif platform.system() == 'Windows':
                self.capture_thread = threading.Thread(target=self.camera_capture_func, daemon=True)
        else:
            self.capture_thread = threading.Thread(target=self.video_capture_func, args=(path,), daemon=True)

        self.capture_thread.start()

        return

    def buffer_ctrl_func(self, path):
        rec_queue = queue.Queue(maxsize=100)
        while True:
            packet = self.stream_queue.get()
            if self.video_stream_info and packet.stream == self.video_stream_info:
                self.latest_stream_queue.put(packet)

            if rec_queue.full():
                _ = rec_queue.get()
            rec_queue.put(packet)

            if self.rec_start_event.is_set():
                out_container = self.make_rec_container()
                rec_proc = threading.Thread(target=self.rec_proc_func, args=(out_container, rec_queue))
                rec_proc.start()
                self.rec_start_event.clear()

            if self.rec_stop_event.is_set():
                rec_queue.put(None)
                self.rec_stop_event.clear()

        return

    def rec_proc_func(self, out_container, rec_queue):
        v_stream = out_container.add_stream_from_template(self.video_stream_info) if self.video_stream_info else None
        a_stream = out_container.add_stream_from_template(self.audio_stream_info) if self.audio_stream_info else None
        v_stream.width = self.video_stream_info.width
        v_stream.height = self.video_stream_info.height
        v_stream.pix_fmt = self.video_stream_info.pix_fmt
        v_stream.time_base = self.video_stream_info.time_base
        v_stream.bit_rate = self.video_stream_info.bit_rate
        a_stream.layout = self.audio_stream_info.layout
        a_stream.format = self.audio_stream_info.format
        a_stream.time_base = self.audio_stream_info.time_base
        v_start_pts = None
        a_start_pts = None

        while True:
            packet = rec_queue.get()
            if packet is None:
                break

            if packet.dts is None or packet.pts is None:
                continue

            if v_start_pts is None:
                if self.video_stream_info and packet.stream == self.video_stream_info and packet.is_keyframe:
                    v_start_pts = packet.pts
                    if self.audio_stream_info:
                        a_start_pts = round(packet.pts * self.video_stream_info.time_base /
                                            self.audio_stream_info.time_base)
                else:
                    continue

            if self.video_stream_info and packet.stream == self.video_stream_info:
                if v_start_pts is None:
                    v_start_pts = packet.pts
                packet.dts -= v_start_pts
                packet.pts -= v_start_pts
                packet.stream = v_stream

            if self.audio_stream_info and packet.stream == self.audio_stream_info:
                if a_start_pts is None:
                    a_start_pts = packet.pts
                packet.dts -= a_start_pts
                packet.pts -= a_start_pts
                packet.stream = a_stream

            packet.dts = max(0, packet.dts)
            packet.pts = max(packet.dts, packet.pts)

            out_container.mux(packet)

        out_container.close()

        return

    def rec_test(self, event):
        if not self.rec:
            self.rec = True
            self.rec_start_event.set()
            print("🟧 録画を開始しました。")
        else:
            if self.rec:
                self.rec = False
                self.rec_stop_event.set()
                print("🟨 録画を終了しました。")

        return

    def img_canvas_make(self):
        self.img_canvas = tk.Canvas(self.root, highlightbackground="#000000", highlightcolor="#000000")
        self.img_canvas.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        blank = np.zeros(BLANK_DUMMY_SIZE, dtype=np.uint8)
        image_pil = Image.fromarray(blank)
        self.image_tk = ImageTk.PhotoImage(image_pil)
        self.imageItemID = self.img_canvas.create_image(0, 0, image=self.image_tk, anchor="nw")

        return

    def image_update_func(self):
        if not self.output_image_queue.empty():
            while not self.output_image_queue.empty():
                image = self.output_image_queue.get()
            rgbimg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.image_update_proc(rgbimg)
        self.root.after(10, self.image_update_func)

    def image_update_proc(self, image):
        image_pil = Image.fromarray(image)
        canvas_width = self.img_canvas.winfo_width()
        canvas_height = self.img_canvas.winfo_height()
        image_pil = ImageOps.pad(image_pil, (canvas_width, canvas_height))
        self.image_tk = ImageTk.PhotoImage(image_pil)
        self.img_canvas.itemconfigure(self.imageItemID, image=self.image_tk)

    def inference_func(self):
        from trt.tensorrt_inferencer import TensorRTInferencer
        from yolox import postprocess
        from visualization.classes import CLASS_NAME_MAP
        from visualization.visualize_detection import DetectionVisualizer

        engine_path = "model/yolox_tiny_20251205_op18.trt"
        score_threshold = 0.1
        output_shape = (1, 151200, 15)
        trt_inferencer = None

        try:
            trt_inferencer = TensorRTInferencer(engine_path)
            visualizer = DetectionVisualizer()

            while True:
                if not self.capture_image_queue.empty():
                    while not self.capture_image_queue.empty():
                        image = self.capture_image_queue.get()
                else:
                    image = self.capture_image_queue.get()
                if image is None:
                    break

                image_height, image_width, _ = image.shape
                input_image = image.transpose(2, 0, 1)
                detected_info = trt_inferencer.infer(
                    inputs=input_image,
                    output_shape=output_shape
                )
                boxes, scores, class_ids = postprocess.get_detection_result(
                    inference_output=detected_info,
                    input_shape=(image_height, image_width),
                    ratio=1
                )

                out_image = image.copy()
                for i in range(len(boxes)):
                    box = boxes[i]
                    score = scores[i]
                    class_id = int(class_ids[i])
                    if score < score_threshold:
                        continue
                    x_min = round(box[0])
                    y_min = round(box[1])
                    x_max = round(box[2])
                    y_max = round(box[3])

                    visualizer.overlay_boxes(
                        image=out_image,
                        left_top=(x_min, y_min),
                        right_bottom=(x_max, y_max),
                        score=score,
                        class_id=class_id,
                        class_name_map=CLASS_NAME_MAP,
                    )

                self.output_image_queue.put(out_image)
        except Exception as e:
            print(f"予期しないエラーが発生しました:\n{e}")
        finally:
            if trt_inferencer is not None:
                trt_inferencer.destroy()

        return
if __name__ == "__main__":
    if 1:
        gui = GUI(path="D:/_LocalUse/unsafe_act_monitoring/input/R0010034.MP4")
    else:
        if len(sys.argv) >= 2 and os.path.isfile(sys.argv[-1]):
            cap = cv2.VideoCapture(sys.argv[-1])
            ret, _ = cap.read()
            if ret:
                gui = GUI(sys.argv[-1])
        else:
            gui = GUI()
