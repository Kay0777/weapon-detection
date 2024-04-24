import cv2
import json
import requests
from queue import Queue
from numpy import ndarray

from src.utils import (
    Create_Output_Filenames,
    File_Convert_To_Base64,
    Compress_Video,
)

from typing import Union, Any
from config import CONF


def Send_Sos_Request_To_Server(url: str, data: dict, timeout: int) -> None:
    try:
        requests.post(
            url=url,
            json=data,
            timeout=timeout)
    except requests.exceptions.Timeout:
        print("Request timed out after 1 second")
    except requests.exceptions.RequestException as e:
        print("Request failed:", e)


def Transform_Point(point: tuple[int, int, int, int], from_shape: tuple[int, int], to_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = point

    from_width, from_height = from_shape
    to_width, to_height = to_shape

    dw = to_width / from_width
    dh = to_height / from_height

    return (
        int(x1 * dw),
        int(y1 * dh),
        int(x2 * dw),
        int(y2 * dh),
    )


def Video_Creator_v2(video_create_task: Queue, sos_tasks: Queue) -> None:
    fps = CONF['FPS']
    bitrate = CONF['BITRATE']
    video_shape = CONF['VIDEO_SHAPE']
    color = CONF['PERSON_WITH_WEAPON']

    while True:
        if video_create_task.empty():
            continue

        task: Union[dict, None] = video_create_task.get()
        if task is None:
            break

        camera: str = task['camera']
        camera_fps: int = task['camera_fps']
        camera_shape: tuple[int, int] = task['camera_shape']
        frames: list[tuple[int, ndarray]] = task['frames']
        people: list[dict[str, Any]] = task['people']

        (
            emit_image_path,
            compressed_video_path,
            real_video_path,
            emit_time
        ) = Create_Output_Filenames(camera=camera)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(
            filename=real_video_path,
            fourcc=fourcc,
            fps=camera_fps,
            frameSize=video_shape)

        frame_index, _ = frames[0]
        suspicious_people: dict = people.pop(0)
        if frame_index > suspicious_people['frame_id']:
            while True:
                if suspicious_people == suspicious_people['frame_id']:
                    break

                if len(people):
                    suspicious_people = people.pop(0)

        image_saved = False
        for frame_index, frame_data in frames:
            if frame_index == suspicious_people['frame_id']:
                for person in suspicious_people['data']:
                    x1, y1, x2, y2 = Transform_Point(
                        point=person,
                        from_shape=camera_shape,
                        to_shape=video_shape)

                    cv2.rectangle(
                        img=frame_data,
                        pt1=(x1, y1),
                        pt2=(x2, y2),
                        color=color,
                        thickness=2,
                        lineType=2)
                    if not image_saved:
                        cv2.imwrite(filename=emit_image_path, img=frame_data)
                        image_saved = True
                if len(people):
                    suspicious_people = people.pop(0)
            videoWriter.write(image=frame_data)

        if videoWriter.isOpened():
            videoWriter.release()

        print('# $ _______________________________________________________ $ #')
        print('# $ ______________  V I D E O   C R E A T E D _____________ $ #')
        Compress_Video(
            input_video_path=real_video_path,
            output_video_path=compressed_video_path,
            bitrate=bitrate)
        print('# $ _______________________________________________________ $ #')

    sos_tasks.put({
        'camera_ip': camera,
        'the_date': emit_time,
        'event_photo': emit_image_path,
        'source': compressed_video_path,
    })


def Video_Creator_v1(video_create_task: Queue, sos_tasks: Queue) -> None:
    weapon_color = CONF['WEAPON_COLOR']
    fps = CONF['FPS']
    bitrate = CONF['BITRATE']

    while True:
        if video_create_task.empty():
            continue

        task: Union[dict, None] = video_create_task.get()
        if task is None:
            break

        camera: str = task['camera']
        frames: list[tuple[int, ndarray]] = task['frames']
        indeces_with_coors: list[tuple[int, tuple[int, int],
                                       tuple[int, int]]] = task['indeces_with_coors']

        (
            emit_image_path,
            compressed_video_path,
            real_video_path,
            emit_time
        ) = Create_Output_Filenames(camera=camera)

        video_shape = frames[0][1].shape[:2][::-1]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(
            filename=real_video_path,
            fourcc=fourcc,
            fps=fps,
            frameSize=video_shape)

        current_index = 0
        image_saved = False
        for frame_index, frame_data in frames:
            if current_index < len(indeces_with_coors) and frame_index == indeces_with_coors[current_index][0]:
                _, pt1, pt2 = indeces_with_coors[current_index]

                cv2.rectangle(
                    img=frame_data,
                    pt1=pt1,
                    pt2=pt2,
                    color=weapon_color,
                    thickness=2,
                    lineType=2)

                if not image_saved:
                    cv2.imwrite(filename=emit_image_path, img=frame_data)
                    image_saved = True
                current_index += 1
            videoWriter.write(image=frame_data)

        if videoWriter.isOpened():
            videoWriter.release()

        print('# $ _______________________________________________________ $ #')
        print('# $ ______________  V I D E O   C R E A T E D _____________ $ #')
        Compress_Video(
            input_video_path=real_video_path,
            output_video_path=compressed_video_path,
            bitrate=bitrate)
        print('# $ _______________________________________________________ $ #')

    sos_tasks.put({
        'camera_ip': camera,
        'the_date': emit_time,
        'event_photo': emit_image_path,
        'source': compressed_video_path,
    })


def Sos_Sender(sos_tasks: Queue) -> None:
    alarm_url, timeout = CONF['ALARM_URL'], CONF['TIMEOUT']

    while True:
        if sos_tasks.empty():
            continue

        data: Union[dict, None] = sos_tasks.get()
        if data is None:
            break

        data.update({
            "event_photo": File_Convert_To_Base64(filename=data['event_photo']),
            "source": File_Convert_To_Base64(filename=data['source'])})

        Send_Sos_Request_To_Server(
            url=alarm_url,
            data=data,
            timeout=timeout
        )
