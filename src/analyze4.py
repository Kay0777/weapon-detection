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

from src.utils import Get_Until_This_Frame_ID

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

        __device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
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
        self.__suspicious_people: dict = {}
        self.__tracker: dict = {}

        self.__track_frame_info: dict = {}

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
        for index, (frame_index, _) in enumerate(self.__frames):
            if frame_index <= frame_from:
                _min = max(_min, index)

            if frame_to <= frame_index:
                _max = index + 1
                break
        video_frames = self.__frames[_min:_max]
        return video_frames

    def __cut_image(self, frame: ndarray, x1: int, y1: int, x2: int, y2: int) -> ndarray:
        return self.transform(
            Image.fromarray(
                obj=cv2.cvtColor(
                    src=frame[y1:y2, x1:x2],
                    code=cv2.COLOR_BGR2RGB
                ))).unsqueeze(0)

    def __draw(self, source: ndarray, pts: list[tuple[int, int]], color: tuple[int, int, int], text: Union[None, str] = None) -> None:
        for x1, y1, x2, y2 in pts:
            cv2.rectangle(
                img=source, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=2, lineType=2)
            if text is not None:
                cv2.putText(
                    img=source, text=text, org=(x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.9, color=color, thickness=5)

    def write_message(self, message: str, mtype: str = 'info') -> None:
        current_time = time.time()
        item = (mtype, f'{message} time: {current_time}')
        self.__errors_and_info_handle_task.put(item=item)

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

        # cap = cv2.VideoCapture('DJI_0033.MOV')

        width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.shape = (width, height)
        self.fps = round(cap.get(cv2.CAP_PROP_FPS))
        self.average_fps_in_frame_read = self.fps

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
            if len(self.__frames) == 2000:
                self.__frames.pop(0)
            self.__frames.append((frame_index, _frame))
            self.frame_queue.put((frame_index, frame))

        self.frame_queue.put((-1, None))
        cap.release()

    def analyze(self):
        EMIT_BATCH_SIZE: int = 5
        EMIT_ON_TIME: int = 2

        print('Analysis is waiting untill camera is connected!')
        self.__event.wait()
        frame_counter: int = 0
        min_frame_index: int = 1
        print(f'Analysis is started! The camera:  {self.camera}')
        while True:
            if self.frame_queue.empty():
                time.sleep(0.1)
                continue

            frame_index, frame_data = self.frame_queue.get()
            if frame_data is None:
                break

            current_time = time.time()
            self.__timestamps_in_analyze.append(current_time)
            if len(self.__timestamps_in_analyze) > 5:
                average_fps_in_analyze = (len(self.__timestamps_in_analyze) - 1) // \
                    (self.__timestamps_in_analyze[-1] -
                     self.__timestamps_in_analyze[0])
                self.__timestamps_in_analyze.clear()

                self.write_message(
                    message=f'Info on Analysis Frame; StreamProcessor FPS average_fps_in_frame_read {average_fps_in_analyze} for {self.camera}')

            results: Results = self.model.track(
                source=frame_data,
                stream=True,
                persist=True,
                verbose=False,
                imgsz=960,
                conf=0.4,
                iou=0.45,
                tracker='botsort.yaml')
            if not results:
                continue

            # WORK WITH RESULTS
            for result in results:
                # FILTER PEOPLE FROM DETECTED OBJECTS
                detected_people = [
                    box for box in result.boxes if int(box.cls.item()) == 2]

                # continue
                if not detected_people:
                    self.write_message(
                        message=f'Camera: {self.camera} Frame ID: {frame_index}; People not detected!')
                    continue

                # print(frame_index, 'detected_people', len(detected_people), [
                #       int(box.id.item()) for box in detected_people])

                detected_people_on_frame_info: dict = {}
                for box in detected_people:
                    if box.id is not None and box.id.item():
                        x1, y1, x2, y2 = self.__box_args(
                            box=box, only_coor=True)
                        person_id: int = int(box.id.item())
                        detected_people_on_frame_info[person_id] = {
                            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, }
                if detected_people_on_frame_info:
                    self.__track_frame_info[frame_index] = detected_people_on_frame_info

                # FILTER WEAPONS FROM DETECTED OBJECTS { RIFLE ID: [0] AND PISTOL ID: [1]}
                detected_weapons = [(box, *self.__box_args(box=box, only_coor=True))
                                    for box in result.boxes if int(box.cls.item()) != 2]
                # __________________________________________________________________________________________________________________________________________________________
                if not detected_weapons:
                    if self.__tracker:
                        for box in detected_people:
                            if box.id is not None and box.id.item() and int(box.id.item()) in self.__tracker:
                                person_id: int = int(box.id.item())
                                detected_on_frames: list = self.__tracker[person_id
                                                                          ]['detected_on_frames']
                                detected_on_frames.append(frame_index)
                    self.write_message(
                        message=f"Camera: {self.camera} Frame ID: {frame_index}; Weapon not detected!")
                    continue
                # __________________________________________________________________________________________________________________________________________________________

                # CROP DETECTED WEAPONS AND TRANSFORM TO IMAGE AND CONCATE IMAGE ON TORCH AND SEND TO DEVICE [CPU OR GPU]
                # _______________________________________________________________________________________________________
                batch_detected_weapon_tensors = torch.cat([
                    self.__cut_image(frame=frame_data,
                                     x1=x1, y1=y1, x2=x2, y2=y2)
                    for _, x1, y1, x2, y2 in detected_weapons
                ]).to(self.device)
                # _______________________________________________________________________________________________________

                with torch.no_grad():
                    batch_input = self.clmodel(batch_detected_weapon_tensors)
                    batch_probabilities = torch.nn.functional.softmax(
                        input=batch_input,
                        dim=1)
                    top1_indices = torch.argmax(
                        input=batch_probabilities,
                        dim=1)

                    for i, top1_index in enumerate(top1_indices):
                        # IF DETECTED OBJECT IS NOT WEAPON LOOP CONTINUE TO NEXT DETECTED DATA
                        if self.class_names[top1_index] != "weapon":
                            continue

                        # GET DETECTED WEAPON BOXES INFO
                        box_1, *_ = detected_weapons[i]
                        # FILTER INTERSECTED PEOPLE
                        intersecting_detected_people = [box_2
                                                        for box_2 in detected_people
                                                        if self.boxes_intersect(box_1.xyxy[0], box_2.xyxy[0])]
                        # IF NOT INTERSECTED PEOPLE LOOP CONTINUE TO NEXT DETECTED DATA
                        if not intersecting_detected_people:
                            continue

                        # GET CLOSEST SUSPICIOUS PERSON
                        closest_person: Boxes = min(intersecting_detected_people,
                                                    key=lambda box_2: (box_1.xyxy[0] - box_2.xyxy[0]).pow(2).sum())
                        if closest_person.id is None:
                            continue

                        # # APPEND CLOSEST SUSPICIOUS PERSON
                        closest_person_id = int(closest_person.id.item())
                        if closest_person_id not in self.__tracker:
                            self.write_message(
                                message=f"Person with weapon detected {closest_person_id}; Camera: {self.camera}")
                            self.__tracker[closest_person_id] = {
                                'counter':  1,
                                'is_alarmable': False,
                                'is_sos_sent': False,
                                'frame_counter': frame_counter,
                                'detected_frame_id': frame_index,
                                'frame_from': 0,
                                'frame_to': 0,
                                'detected_on_frames': [frame_index],
                            }
                            print('Added:', closest_person_id)

                        elif self.__tracker[closest_person_id]['counter'] < EMIT_BATCH_SIZE - 1:
                            self.write_message(
                                message=f"Person with weapon detected {closest_person_id}; Camera: {self.camera}")
                            person: dict = self.__tracker[closest_person_id]
                            person['counter'] += 1
                            person['detected_on_frames'].append(frame_index)

                        elif self.__tracker[closest_person_id]['counter'] == EMIT_BATCH_SIZE:
                            person: dict = self.__tracker[closest_person_id]
                            person['detected_on_frames'].append(frame_index)

                        elif (frame_counter - self.__tracker[closest_person_id]['frame_counter']) / self.fps <= EMIT_ON_TIME:
                            self.write_message(
                                message=f"Alarm is confirmed {closest_person_id}; Camera: {self.camera}")

                            self.__suspicious_people_ids.add(closest_person_id)

                            person: dict = self.__tracker[closest_person_id]
                            frame_from, frame_to = Get_Until_This_Frame_ID(
                                alarm_frame_id=person['detected_frame_id'],
                                camera_fps=max(CONF['FPS'], min(int(self.fps), int(self.average_fps_in_frame_read))))
                            person.update(
                                {'frame_from': frame_from, 'frame_to': frame_to, 'is_alarmable': True})
                            person['counter'] += 1
                            person['detected_on_frames'].append(frame_index)

                            # UPDATE min frame index TO CLEAN STACKED DATA
                            min_frame_index = max(min_frame_index, frame_from)

                        else:
                            self.write_message(
                                message=f"Alarm was not confirmed, {frame_counter}, FPS: {self.fps}; Camera: {self.camera}")
                            self.__tracker.pop(closest_person_id)
            frame_counter += 1
            sos_sent_people = tuple(id for id in self.__tracker.keys()
                                    if self.__tracker[id]['is_sos_sent'] and
                                    self.__tracker[id]['frame_to'] < min_frame_index)
            if sos_sent_people:
                for person_id in sos_sent_people:
                    self.__tracker.pop(person_id)
                    self.__suspicious_people_ids.remove(person_id)

            # CHECK CAMERA IS ALARMABLE
            # CHECK LAST FRAME INDEX OF FRAMES STACK LESS THAN UNTIL FARME INDEX TO CREATE A VIDEO
            is_alarmable_people = tuple(id for id in self.__tracker.keys()
                                        if self.__tracker[id]['is_alarmable'] and
                                        not self.__tracker[id]['is_sos_sent'] and
                                        self.__tracker[id]['frame_to'] <= self.__frames[-1][0])
            for person_id in is_alarmable_people:
                frames = self.__cut_frames(
                    frame_from=frame_from, frame_to=frame_to)
                self.__video_create_task.put({
                    'camera': self.camera,
                    'person': person_id,
                    'camera_fps': max(CONF['FPS'], min(int(self.fps), int(self.average_fps_in_frame_read))),
                    'camera_shape': self.shape,
                    'frames': frames,
                    'suspicious': self.__suspicious_people_ids,
                    'people': self.__track_frame_info
                })
                self.__tracker[person_id]['is_sos_sent'] = True

        del self.__track_frame_info
        del self.__tracker
        print('Analyze is Done!')

    def start(self):
        reader_thread = Thread(target=self.frame_reader)
        analysis_thread = Thread(target=self.analyze)

        reader_thread.start()
        analysis_thread.start()

        reader_thread.join()
        analysis_thread.join()
