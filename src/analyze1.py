import cv2
import torch
from ultralytics import YOLO
from src.depends import MaxSizeQueue
from torchvision import models, transforms


import time
import logging
from PIL import Image
from queue import Queue
from threading import Thread
from collections import deque

from config import CONF

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class StreamProcessor:
    def __init__(self, camera: str, login_passwords: list[tuple[str, str]], video_create_task: Queue) -> None:
        self.camera = camera
        self.login_passwords = login_passwords
        self.save_dir = CONF['WEAPON_IMAGES_SAVE_DIR']

        self.frame_queue = MaxSizeQueue(maxsize=CONF['MAX_QUEUE_SIZE'])
        self.timestamps = deque(maxlen=10)

        isavaible = CONF['DEVICE_TYPE'] == 'GPU' and torch.cuda.is_available()
        __device = 'cuda:0' if isavaible else 'cpu'
        self.device = torch.device(__device)

        self.load_detection_model()
        self.load_classification_model()
        self.craete_transforms()

        self.logger = logging.getLogger(self.__class__.__name__)

        self.__video_create_task: Queue = video_create_task
        self.__frames: list = []
        self.__detected_indeces: list[int] = []

    def load_detection_model(self) -> None:
        weapon_detect_model = CONF['WEAPON_YOLO_MODEL']
        weapon_detect_task = CONF['WEAPON_TASK']

        self.model = YOLO(model=weapon_detect_model, task=weapon_detect_task)

    def load_classification_model(self) -> None:
        weights = models.EfficientNet_B0_Weights.DEFAULT
        self.clmodel = models.efficientnet_b0(weights=weights)

        num_features = self.clmodel.classifier[1].in_features
        self.clmodel.classifier[1] = torch.nn.Linear(num_features, 2)

        weapon_class_model = CONF['WEAPON_CLASSIFY_TORCH_MODEL']
        model = torch.load(f=weapon_class_model)
        self.clmodel.load_state_dict(state_dict=model)
        self.clmodel.to(self.device)
        self.clmodel.eval()

    def craete_transforms(self) -> None:
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])

    def create_rtsp_url(self, login: str, password: str) -> str:
        return f'rtsp://{login}:{password}@{self.camera}:554/cam/realmonitor?channel=1&subtype=0'

    def frame_reader(self) -> None:
        connected = False
        for login, password in self.login_passwords:
            cap = cv2.VideoCapture(self.create_rtsp_url(
                login=login, password=password
            ))
            if cap.isOpened():
                connected = True
                self.logger.info(
                    f"Successfully opened RTSP stream with login: {login}; password: {password}")
                break
            self.logger.warning(
                f"Failed to open RTSP stream with login: {login}; password: {password}")

        if not connected:
            self.logger.error("All logins and passwords attempts failed!")
            return

        width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)

        frame_index = 0
        scale = CONF['SCALE']
        while True:
            ret, frame = cap.read()
            if not ret:
                self.logger.error("Failed to read frame from the camera.")
                break

            frame_index += 1
            _frame = cv2.resize(
                src=frame,
                dsize=(width // scale, height // scale)
            )

            self.__frames.append((frame_index, _frame))
            self.frame_queue.put((frame_index, frame))
        cap.release()

    def analyze(self):
        detected_frames_count = CONF['DETECTED_FRAMES_COUNT']
        waiting_frames_count = CONF['FPS'] * CONF['VIDEO_LENGTH']
        scale = CONF['SCALE']
        class_names = CONF['CLASSNAMES']

        image_counter = 0
        while True:
            if self.frame_queue.empty():
                time.sleep(0.01)
                continue

            frame_index, frame_data = self.frame_queue.get()

            current_time = time.time()
            self.timestamps.append(current_time)
            if len(self.timestamps) > 1:
                average_fps = (
                    len(self.timestamps) - 1) // (self.timestamps[-1] - self.timestamps[0])
                # self.logger.info(f"FPS: {average_fps} for {self.camera}")

            results = self.model.predict(
                source=frame_data,
                verbose=False,
                imgsz=960,
                conf=0.4,
                iou=0.6)
            if not results:
                continue

            detected_objects = [
                tuple(int(point) for point in box.xyxy[0].tolist())
                for r in results
                for box in r.boxes
                if any(tuple(box.xyxy[0].tolist()))
            ]

            if not detected_objects:
                continue

            batch_images = [
                self.transform(
                    Image.fromarray(
                        cv2.cvtColor(
                            frame_data[y1:y2, x1:x2],
                            cv2.COLOR_BGR2RGB
                        )
                    )
                ).unsqueeze(0)
                for x1, y1, x2, y2 in detected_objects
            ]

            batch_tensor = torch.cat(batch_images).to(self.device)

            with torch.no_grad():
                batch_output = self.clmodel(batch_tensor)
                batch_probabilities = torch.nn.functional.softmax(
                    batch_output, dim=1)
                top1_indices = torch.argmax(batch_probabilities, dim=1)

                for i, top1_index in enumerate(top1_indices):
                    if class_names[top1_index] == "weapon":
                        # image_counter += 1
                        # box, cropped_image = detected_objects[i]
                        # pil_image = Image.fromarray(
                        #     cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
                        # save_path = os.path.join(
                        #     self.save_dir, f"{self.camera}_{image_counter}_weapon.jpg")
                        # pil_image.save(save_path)

                        # # Append detected frames info {Frame ID and Coordinates}
                        # x1, y1, x2, y2 = box
                        x1, y1, x2, y2 = detected_objects[i]
                        self.__detected_indeces.append(
                            (frame_index, (x1 // scale, y1 // scale), (x2 // scale, y2 // scale)))

            if len(self.__detected_indeces) >= detected_frames_count and len(self.__frames) >= waiting_frames_count:
                self.__video_create_task.put({
                    'camera': self.camera,
                    'frames': self.__frames,
                    'indeces_with_coors': self.__detected_indeces,
                })
                self.__detected_indeces.clear()
                self.__frames.clear()

    def start(self):
        reader_thread = Thread(target=self.frame_reader)
        analysis_thread = Thread(target=self.analyze)

        reader_thread.start()
        analysis_thread.start()

        reader_thread.join()
        analysis_thread.join()
