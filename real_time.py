import os
import cv2
import time
import torch
import shutil
import logging
import threading
import multiprocessing
from PIL import Image
from queue import Queue
from ultralytics import YOLO
from collections import deque
from torchvision import models, transforms, ops
from turbojpeg import TurboJPEG, TJPF_BGR, TJSAMP_444
# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class StreamProcessor:
    def __init__(self, camera, passwords, save_dir='save/'):
        self.camera = camera
        self.passwords = passwords
        self.save_dir = save_dir
        self.image_counter = 0
        self.frame_counter = 0
        self.frame_queue = Queue(maxsize=2)
        self.resized_frame_queue = Queue(maxsize=2)
        self.timestamps = deque(maxlen=10)
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.load_detection_model()
        self.clmodel = self.load_classification_model()
        self.transform = self.get_transforms()
        # self.class_labels = {0: 'rifle', 1: 'pistol', 2: 'person'}
        self.class_labels = {0: 'rifle', 1: 'pistol', 2: 'person'}
        self.class_names = ["others", "weapon"]
        self.logger = logging.getLogger(self.__class__.__name__)
        self.last_class_2_track_id = []
        self.weapon_counter = {}
        self.person_counter = {}
        self.weapon_frame = {}
        self.fps = 0
        self.beginning = time.time()
        self.jpeg = TurboJPEG(
            'D:/Weapon_Project/libjpeg-turbo64/bin/turbojpeg.dll')

    def load_detection_model(self):
        model = YOLO('weights/best.pt', task='detect')
        # model = YOLO('1024.pt', task='detect')

        return model

    def load_classification_model(self):
        weights = models.EfficientNet_B0_Weights.DEFAULT
        clmodel = models.efficientnet_b0(weights=weights)
        num_features = clmodel.classifier[1].in_features
        clmodel.classifier[1] = torch.nn.Linear(num_features, 2)
        clmodel.load_state_dict(torch.load('weapon_classify.pth'))
        clmodel.to(self.device)
        clmodel.eval()
        return clmodel

    def get_transforms(self):
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])

    def draw_box(self, frame, box, color=(0, 0, 255)):
        cv2.rectangle(frame, (int(box.xyxy[0][0]), int(box.xyxy[0][1])), (int(
            box.xyxy[0][2]), int(box.xyxy[0][3])), color, 8)
        conf = box.conf.item()
        track_id = box.id.item()
        label = self.class_labels[int(box.cls.item())]
        cv2.putText(frame, f'ID: {track_id}, {label}, {conf}', (int(box.xyxy[0][0]), int(
            box.xyxy[0][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 5)

    def draw_weapon_box(self, frame, box, color=(0, 0, 255)):
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 5)

    def boxes_intersect(self, box1, box2):
        # Unpack the coordinates
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        # Check for overlap
        return (x1_min < x2_max and x1_max > x2_min and
                y1_min < y2_max and y1_max > y2_min)

    def save_frame(self):
        frame_batch = []
        while True:
            if self.resized_frame_queue.empty():
                time.sleep(0.01)
                continue
            frame = self.resized_frame_queue.get()
            frame_batch.append(frame)
            self.image_counter += 1

            # Check if we have accumulated 10 frames
            if len(frame_batch) != 10:
                continue

            # Calculate the subfolder name based on the image counter
            batch_start = self.image_counter - 10
            batch_end = self.image_counter - 1
            subfolder_name = f'{batch_start}-{batch_end}'

            # Create the directory structure if it doesn't exist
            save_path = os.path.join(
                self.save_dir, self.camera, subfolder_name)
            os.makedirs(save_path, exist_ok=True)

            # Save all frames in the batch
            for i, frame in enumerate(frame_batch):
                file_name = os.path.join(save_path, f'{i}.jpg')
                with open(file_name, 'wb') as outfile:
                    jpeg_image = self.jpeg.encode(
                        frame, pixel_format=TJPF_BGR, jpeg_subsample=TJSAMP_444)
                    outfile.write(jpeg_image)

            # Clear the batch to start accumulating the next batch of frames
            frame_batch = []

            all_subfolders = [os.path.join(self.save_dir, self.camera, d) for d in os.listdir(
                os.path.join(self.save_dir, self.camera))]
            all_subfolders.sort(key=lambda x: os.path.getmtime(x))

            if len(all_subfolders) > 50:
                # Remove the oldest subfolders to maintain only the latest 50
                for folder in all_subfolders[:-50]:
                    shutil.rmtree(folder)
                    self.logger.info(f"Deleted old folder: {folder}")

    def frame_reader(self):
        connected = False
        for password in self.passwords:
            rtsp_url = f'rtsp://admin:{password}@{self.camera}:554/cam/realmonitor?channel=1&subtype=0'
            print(rtsp_url)
            cap = cv2.VideoCapture(rtsp_url)
            if cap.isOpened():
                connected = True
                self.fps = cap.get(cv2.CAP_PROP_FPS)
                self.logger.info(f"FPS of the camera: {self.fps}")
                self.logger.info(
                    f"Successfully opened RTSP stream with password {password}")
                break
            else:
                self.logger.warning(
                    f"Failed to open RTSP stream with password {password}")

        if not connected:
            self.logger.error("All password attempts failed.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                self.logger.error("Failed to read frame from the camera.")
                break
            if self.frame_queue.full():
                self.frame_queue.get()
            self.frame_queue.put(frame)

        cap.release()

    def analyze(self):
        while True:
            if self.frame_queue.empty():
                time.sleep(0.01)
                continue
            frame = self.frame_queue.get()
            current_time = time.time()
            self.timestamps.append(current_time)
            if len(self.timestamps) > 1:
                average_fps = (len(self.timestamps) - 1) / \
                    (self.timestamps[-1] - self.timestamps[0])
                # self.logger.info(f"FPS: {average_fps} for {self.camera}")

            results = self.model.track(frame, stream=True, verbose=False, persist=True,
                                       imgsz=960, conf=0.4, iou=0.45, tracker='bytetrack.yaml')
            for result in results:
                class_0_boxes = [
                    box for box in result.boxes if int(box.cls.item()) != 2]
                class_2_boxes = [
                    box for box in result.boxes if int(box.cls.item()) == 2]

                if not class_0_boxes:
                    if not class_2_boxes and self.last_class_2_track_id is None:
                        continue
                    else:
                        for box in class_2_boxes:
                            if box.id is not None and box.id.item() in self.last_class_2_track_id:
                                self.draw_box(frame, box)
                        continue

                detections = [(box.xyxy[0], frame[int(box.xyxy[0][1]):int(box.xyxy[0][3]), int(
                    box.xyxy[0][0]):int(box.xyxy[0][2])]) for box in class_0_boxes]
                batch_tensor = torch.cat([self.transform(Image.fromarray(cv2.cvtColor(
                    crop, cv2.COLOR_BGR2RGB))).unsqueeze(0) for _, crop in detections]).to(self.device)

                with torch.no_grad():
                    batch_output = self.clmodel(batch_tensor)
                    batch_probabilities = torch.nn.functional.softmax(
                        batch_output, dim=1)
                    top1_indices = torch.argmax(batch_probabilities, dim=1)

                    for i, top1_index in enumerate(top1_indices):
                        self.logger.info(f"{self.class_names[top1_index]}")
                        if self.class_names[top1_index] == "weapon":
                            box, _ = detections[i]

                            if class_2_boxes:
                                intersecting_class_2_boxes = [
                                    box_2 for box_2 in class_2_boxes if self.boxes_intersect(box, box_2.xyxy[0])]
                            else:
                                intersecting_class_2_boxes = []

                            if intersecting_class_2_boxes:
                                closest_box = min(intersecting_class_2_boxes, key=lambda b: (
                                    b.xyxy[0] - box).pow(2).sum())

                                if closest_box.id is None:
                                    continue

                                id = closest_box.id.item()

                                if id not in self.weapon_counter and id not in self.weapon_frame:
                                    self.weapon_counter.setdefault(id, 0)
                                    self.weapon_frame.setdefault(id, 0)

                                if self.weapon_counter[id] == 0:
                                    self.logger.info(
                                        f"Person with weapon detected")
                                    self.weapon_frame[id] = self.frame_counter
                                    self.weapon_counter[id] += 1
                                elif self.weapon_counter[id] < 4:
                                    # self.logger.info(f"Checking for false alarm")
                                    self.weapon_counter[id] += 1
                                elif self.weapon_counter[id] == 5:
                                    self.draw_box(frame, closest_box)
                                elif (self.frame_counter - self.weapon_frame[id]) / self.fps <= 1:
                                    self.logger.info(f"Alarm is confirmed")
                                    self.last_class_2_track_id.append(id)
                                    self.draw_box(frame, closest_box)
                                    self.weapon_counter[id] += 1
                                else:
                                    self.logger.info(
                                        f"Alarm was not confirmed, {self.frame_counter}, {self.weapon_frame[id]}, {self.fps}")
                                    self.weapon_counter.pop(id, None)
                                    self.weapon_frame.pop(id, None)
                            else:
                                self.draw_weapon_box(frame, box)

            self.frame_counter += 1
            resized_frame = cv2.resize(frame, (1280, 720))
            if self.resized_frame_queue.full():
                self.resized_frame_queue.get()
            self.resized_frame_queue.put(resized_frame)

            cv2.imshow(f'{self.camera}', resized_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    def start(self):
        reader_thread = threading.Thread(target=self.frame_reader)
        analysis_thread = threading.Thread(target=self.analyze)
        # save_thread = threading.Thread(target=self.save_frame)
        reader_thread.start()
        analysis_thread.start()
        # save_thread.start()
        reader_thread.join()
        analysis_thread.join()
        # save_thread.join()


def process_cameras(cameras):
    for camera, passwords in cameras:
        threading.Thread(target=main, args=(camera, passwords)).start()


def main(camera, password):
    processor = StreamProcessor(camera, password)
    processor.start()


if __name__ == "__main__":
    camera = [('192.168.4.35', ['smartbase404'])]
    # camera = [('10.144.168.163', ['parol12345'])]

    process_cameras(camera)
    # cpu_count = multiprocessing.cpu_count()
    # process_count = 8  # Number of processes you want to run
    # cameras = [('192.168.4.{}'.format(i), ['smartbase404', 'parol12345']) for i in range(1, 37)]
    # # new_carmeras = [('10.144.132.194', ['parol12345']), ('10.144.132.198', ['parol12345']), ('10.144.132.195', ['123456'])]

    # # print (cameras)
    # # cameras = cameras.extend(new_carmeras)
    # cameras_per_process = len(cameras) // process_count

    # processes = []
    # for i in range(process_count):
    #     start_index = i * cameras_per_process
    #     end_index = start_index + cameras_per_process
    #     camera_subset = cameras[start_index:end_index]

    #     # Create and start a multiprocessing.Process
    #     p = multiprocessing.Process(target=process_cameras, args=(camera_subset,))
    #     processes.append(p)
    #     p.start()

    #     # Optionally set CPU affinity for each process to limit the CPUs they can run on
    #     if hasattr(os, 'sched_setaffinity'):
    #         cpu_subset = range(i * (cpu_count // process_count), (i + 1) * (cpu_count // process_count))
    #         os.sched_setaffinity(p.pid, cpu_subset)

    # for p in processes:
    #     p.join()
