import cv2
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes
from src.depends import MaxSizeQueue
from torchvision import models, transforms
from numpy import ndarray

import time
import logging
from PIL import Image
from queue import Queue
from copy import deepcopy
from threading import Thread, Event
from collections import deque
from typing import Union

from src.utils import Get_Until_This_Frame_ID, SuspiciousPeople

from config import CONF

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class StreamProcessor:
    def __init__(self, camera: str, login_passwords: list[tuple[str, str]], video_create_task: Queue, errors_and_info_handle_task: Queue) -> None:
        self.fps: int = 0
        self.camera: str = camera
        self.shape: Union[None, tuple[int, int]] = None
        self.__event: Event = Event()

        self.login_passwords = login_passwords
        self.save_dir = CONF['WEAPON_IMAGES_SAVE_DIR']

        self.image_counter = 0

        self.frame_queue = MaxSizeQueue(maxsize=2)

        self.average_fps_in_frame_read = None
        self.__timestamps_in_frame_read: list = []
        self.__timestamps_in_analyze: list = []

        isavailable = CONF['DEVICE_TYPE'] == 'GPU' and torch.cuda.is_available()
        __device = 'cuda:0' if isavailable else 'cpu'
        self.device = torch.device(__device)

        self.load_detection_model()
        self.load_classification_model()
        self.craete_transforms()

        self.class_labels = {0: 'rifle', 1: 'pistol', 2: 'person'}
        self.class_names = ["others", "weapon"]

        self.logger = logging.getLogger(self.__class__.__name__)

        self.__video_create_task: Queue = video_create_task
        self.__frames: list = []

        self.__errors_and_info_handle_task: Queue = errors_and_info_handle_task

        self.__suspicious_people_ids: set = set()
        self.__suspicious_people: list[dict] = []
        self.__tracker: dict = {}

    def load_detection_model(self) -> None:
        try:
            weapon_detect_model = CONF['WEAPON_YOLO_MODEL']
            weapon_detect_task = CONF['WEAPON_TASK']

            self.model = YOLO(model=weapon_detect_model,
                              task=weapon_detect_task)
        except Exception:
            self.__errors_and_info_handle_task.put(item=(
                'error', 'Error on Load Detection Model; StreamProcessor def load_detection_model 68 line'))

    def load_classification_model(self) -> None:
        try:
            weights = models.EfficientNet_B0_Weights.DEFAULT
            self.clmodel = models.efficientnet_b0(weights=weights)

            num_features = self.clmodel.classifier[1].in_features
            self.clmodel.classifier[1] = torch.nn.Linear(num_features, 2)

            weapon_class_model = CONF['WEAPON_CLASSIFY_TORCH_MODEL']
            model = torch.load(f=weapon_class_model)
            self.clmodel.load_state_dict(state_dict=model)
            self.clmodel.to(self.device)
            self.clmodel.eval()
        except Exception:
            self.__errors_and_info_handle_task.put(
                item=('error', 'Error on Load Classification Model; StreamProcessor def load_classification_model 74 line'))

    def craete_transforms(self) -> None:
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])

    def create_rtsp_url(self, login: str, password: str) -> str:
        return f'rtsp://{login}:{password}@{self.camera}:554/cam/realmonitor?channel=1&subtype=0'

    def boxes_intersect(self, box1: tuple[int, int, int, int], box2: tuple[int, int, int, int]) -> bool:
        # Unpack the coordinates
        try:
            x1_min, y1_min, x1_max, y1_max = box1
            x2_min, y2_min, x2_max, y2_max = box2

            # Check for overlap
            return (
                x1_min < x2_max and
                x1_max > x2_min and
                y1_min < y2_max and
                y1_max > y2_min)
        except Exception:
            self.__errors_and_info_handle_task.put(item=(
                'error', 'Error on Boxes Intersect; StreamProcessor def boxes_intersect 107 line'))
        return False

    def __box_args(self, box: Boxes, only_coor: bool = False) -> Union[tuple[tuple[int, int, int, int], int, float, int], tuple[int, int, int, int]]:
        _coors = tuple(int(value) for value in box.xyxy[0].tolist())
        if only_coor:
            return _coors

        _track_id = int(box.id.item())
        _conf = round(100 * box.conf.item())
        _label = self.class_labels[int(box.cls.item())]
        return (
            _coors,
            _track_id,
            _conf,
            _label
        )

    def __cut_frames(self, frame_from: int, frame_to: int) -> list[tuple[int, ndarray]]:
        _min, _max = 0, -1
        for _index, _ in self.__frames:
            if _index == frame_from:
                _min = _index - 1
            if _index == frame_to:
                _max = _index
                break
        video_frames = self.__frames[_min:_max]
        self.__frames = self.__frames[_min:]
        return video_frames

    def __draw(self, source: ndarray, pts: list[tuple[int, int]], color: tuple[int, int, int], text: Union[None, str] = None) -> None:
        for x1, y1, x2, y2 in pts:
            cv2.rectangle(
                img=source, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=2, lineType=2)
            if text is not None:
                cv2.putText(
                    img=source, text=text, org=(x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.9, color=color, thickness=5)

    def frame_reader(self):
        connected = False
        for login, password in self.login_passwords:
            cap = cv2.VideoCapture(self.create_rtsp_url(
                login=login,
                password=password
            ))

            if cap.isOpened():
                connected = True
                current_time = time.time()
                item = (
                    'info', f"Successfully opened RTSP stream with login: {login}; password: {password}! The camera: {self.camera} time: {current_time}")
                self.__errors_and_info_handle_task.put(item=item)
                break
            current_time = time.time()
            item = (
                'error', f"Failed to open RTSP stream with login: {login}; password: {password}! The camera: {self.camera} time: {current_time}")
            self.__errors_and_info_handle_task.put(item=item)

        if not connected:
            current_time = time.time()
            item = (
                'error', f"All logins and passwords attempts failed! The camera: {self.camera} time: {current_time}")
            self.__errors_and_info_handle_task.put(item=item)
            return

        width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.shape = (width, height)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.average_fps_in_frame_read = self.fps

        scale = CONF['SCALE']
        reshape = CONF['VIDEO_SHAPE']

        # RUN WAITING THREADS
        self.__event.set()
        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                current_time = time.time()
                item = (
                    'error', f"Failed to read frame from the camera: {self.camera} time: {current_time}")
                self.__errors_and_info_handle_task.put(item=item)
                break
            frame_index += 1

            current_time = time.time()
            self.__timestamps_in_frame_read.append(current_time)
            if len(self.__timestamps_in_frame_read) > 5:
                self.average_fps_in_frame_read = (len(self.__timestamps_in_frame_read) - 1) // \
                    (self.__timestamps_in_frame_read[-1] -
                     self.__timestamps_in_frame_read[0])
                self.__timestamps_in_frame_read.clear()

                item = (
                    'info', f'Info on Reading Frame From Camera; StreamProcessor FPS average_fps_in_frame_read {self.average_fps_in_frame_read} for {self.camera} time: {current_time}')
                self.__errors_and_info_handle_task.put(item=item)

            _frame = cv2.resize(src=frame, dsize=reshape)
            self.__frames.append((frame_index, _frame))
            self.frame_queue.put((frame_index, frame))
        cap.release()

    def analyze(self):
        EMIT_BATCH_SIZE: int = 5
        EMIT_ON_TIME: int = 1

        alarm_interval: int = CONF['ALARM_INTERVAL']
        min_frame_index: float = float('inf')

        frame_from, frame_to = -1, -1
        frame_counter: int = 0
        is_alarmable: bool = False
        last_alarmed_time: Union[None, float] = None
        print('Analysis is waiting untill camera is connected!')
        self.__event.wait()
        print('Analysis is started!')

        while True:
            if self.frame_queue.empty():
                continue
            frame_index, frame_data = self.frame_queue.get()

            # print('in Analyze Frame ID:', frame_index)
            # continue

            current_time = time.time()
            self.__timestamps_in_analyze.append(current_time)
            if len(self.__timestamps_in_analyze) > 5:
                average_fps_in_analyze = (len(self.__timestamps_in_analyze) - 1) // \
                    (self.__timestamps_in_analyze[-1] -
                     self.__timestamps_in_analyze[0])
                self.__timestamps_in_analyze.clear()

                item = (
                    'info', f'Info on Analysis Frame; StreamProcessor FPS average_fps_in_frame_read {average_fps_in_analyze} for {self.camera} time: {current_time}')
                self.__errors_and_info_handle_task.put(item=item)

            results: Results = self.model.track(
                source=frame_data,
                stream=True,
                persist=True,
                verbose=False,
                imgsz=960,
                conf=0.4,
                iou=0.45,
                tracker='bytetrack.yaml')
            if not results:
                continue

            # FULL INFO OF CURRENT ANALYZED FRAME INFO
            info = {'frame_id': frame_index, 'data': []}
            for result in results:
                # FILTER PEOPLE FROM DETECTED OBJECTS
                detected_people = [
                    box for box in result.boxes if int(box.cls.item()) == 2]
                if not detected_people:
                    current_time = time.time()
                    item = (
                        'info', f'Camera: {self.camera} Frame ID: {frame_index}; People not detected! time: {current_time}')
                    self.__errors_and_info_handle_task.put(item=item)
                    continue

                # FILTER WEAPONS FROM DETECTED OBJECTS { RIFLE ID: [0] AND PISTOL ID: [1]}
                detected_weapons = [(box, *self.__box_args(box=box, only_coor=True))
                                    for box in result.boxes if int(box.cls.item()) != 2]
                if not detected_weapons:
                    if self.__suspicious_people_ids:
                        detected_suspicious_people = [
                            self.__box_args(box=box, only_coor=True)
                            for box in detected_people
                            if box.id is not None and box.id.item() in self.__suspicious_people_ids]
                        info.update({'data': detected_suspicious_people})
                        self.__suspicious_people.append(info)

                    current_time = time.time()
                    item = (
                        'info', f'Camera: {self.camera} Frame ID: {frame_index}; Weapon not detected! time: {current_time}')
                    self.__errors_and_info_handle_task.put(item=item)
                    continue

                # CROP DETECTED WEAPONS AND TRANSFORM TO IMAGE AND CONCATE IMAGE ON TORCH AND SEND TO DEVICE [CPU OR GPU]
                # _______________________________________________________________________________________________________
                batch_detected_weapon_tensors = torch.cat([
                    self.transform(
                        Image.fromarray(
                            obj=cv2.cvtColor(
                                src=frame_data[y1:y2, x1:x2],
                                code=cv2.COLOR_BGR2RGB
                            ))).unsqueeze(0)
                    for _, x1, y1, x2, y2 in detected_weapons
                ]).to(self.device)
                # _______________________________________________________________________________________________________

                data = []
                with torch.no_grad():
                    batch_input = self.clmodel(batch_detected_weapon_tensors)
                    batch_probabilities = torch.nn.functional.softmax(
                        input=batch_input,
                        dim=1)
                    top1_indices = torch.argmax(
                        input=batch_probabilities,
                        dim=1)

                    msg = ''
                    for i, top1_index in enumerate(top1_indices):
                        # IF DETECTED OBJECT IS NOT WEAPON LOOP CONTINUE TO NEXT DETECTED DATA
                        if self.class_names[top1_index] != "weapon":
                            continue

                        # GET DETECTED WEAPON BOXES INFO
                        box, *_ = detected_weapons[i]
                        # FILTER INTERSECTED PEOPLE
                        intersecting_detected_people = [box_2
                                                        for box_2 in detected_people
                                                        if self.boxes_intersect(box.xyxy[0], box_2.xyxy[0])]
                        # IF NOT INTERSECTED PEOPLE LOOP CONTINUE TO NEXT DETECTED DATA
                        if not intersecting_detected_people:
                            continue

                        # GET CLOSEST SUSPICIOUS PERSON
                        closest_person: Boxes = min(intersecting_detected_people,
                                                    key=lambda _box: (_box.xyxy[0] - box.xyxy[0]).pow(2).sum())
                        if closest_person.id is None:
                            continue

                        # APPEND CLOSEST SUSPICIOUS PERSON
                        data.append(
                            self.__box_args(
                                box=closest_person,
                                only_coor=True))

                        closest_person_id = int(closest_person.id.item())
                        if closest_person_id not in self.__tracker:
                            msg = f"Person with weapon detected {closest_person_id}"
                            self.__tracker[closest_person_id] = {
                                'counter':  1,
                                'frame_counter': frame_counter,
                            }

                        elif self.__tracker[closest_person_id]['counter'] < EMIT_BATCH_SIZE - 1:
                            msg = f"Person counter is updated! {closest_person_id}"
                            self.__tracker[closest_person_id]['counter'] += 1

                        elif self.__tracker[closest_person_id]['counter'] == EMIT_BATCH_SIZE:
                            # print('self.__suspicious_people_ids',
                            #       self.__suspicious_people_ids)
                            pass

                        elif (frame_counter - self.__tracker[closest_person_id]['frame_counter']) / self.fps <= EMIT_ON_TIME:
                            msg = f"Alarm is confirmed {closest_person_id}"
                            is_alarmable = True

                            self.__suspicious_people_ids.add(closest_person_id)
                            self.__tracker[closest_person_id]['counter'] += 1
                            min_frame_index = min(frame_index, min_frame_index)

                        else:
                            msg = f"Alarm was not confirmed, {frame_counter}, FPS: {self.fps}"
                            if closest_person_id in self.__suspicious_people_ids:
                                self.__suspicious_people_ids.remove(
                                    closest_person_id)
                            self.__tracker.pop(closest_person_id)
                            self.__suspicious_people.clear()

                        current_time = time.time()
                        item = (
                            'info',
                            f'{msg} time: {current_time}'
                        )
                        self.__errors_and_info_handle_task.put(item=item)
                if data:
                    info.update({'data': data})
                    self.__suspicious_people.append(info)
            frame_counter += 1
            current_time = time.monotonic()

            # CHECK CAMERA IS ALARMABLE
            # CHECK LAST FRAME INDEX OF FRAMES STACK LESS THAN UNTIL FARME INDEX TO CREATE A VIDEO
            if is_alarmable and (last_alarmed_time is None or alarm_interval <= current_time - last_alarmed_time):
                # ______________________________________________________
                if frame_from == -1 and frame_to == -1:
                    frame_from, frame_to = Get_Until_This_Frame_ID(
                        alarm_frame_id=min_frame_index)

                if frame_to <= self.__frames[-1][0]:
                    frame = self.__cut_frames(
                        frame_from=frame_from, frame_to=frame_to)
                    if frame:
                        self.__video_create_task.put({
                            'camera': self.camera,
                            'camera_fps': self.average_fps_in_frame_read,
                            'camera_shape': self.shape,
                            'frames': frame,
                            'people': self.__suspicious_people
                        })

                        frame_from, frame_to = -1, -1
                        self.__suspicious_people.clear()
                        min_frame_index = float('inf')
                        is_alarmable = False
                        last_alarmed_time = current_time

    def start(self):
        reader_thread = Thread(target=self.frame_reader)
        analysis_thread = Thread(target=self.analyze)

        reader_thread.start()
        analysis_thread.start()

        reader_thread.join()
        analysis_thread.join()
