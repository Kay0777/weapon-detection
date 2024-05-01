import cv2
import json
import time
import requests
from queue import Queue
from numpy import ndarray
from threading import Thread

from src.utils import (
    Create_Output_Filenames,
    File_Convert_To_Base64,
    Compress_Video,
)

from src.tools import (
    Cut_New_Coor,
    Transform_Point,
    Interpolate_Person_Coordinates_V1,
    Interpolate_Person_Coordinates_V2,
    Create_Interpolate_Person_Coors,
    interpolate_person_coordinates
)

from typing import Union, Any
from config import CONF

VIDEO_CREATE_COUNTER: int = 10


def Send_Sos_Request_To_Server(url: str, data: dict, timeout: int) -> None:
    try:
        requests.post(
            url=url,
            json=data,
            timeout=timeout)
        print('Alarm is sent!')
    except requests.exceptions.Timeout:
        print("Request timed out after 1 second")
    except requests.exceptions.RequestException as e:
        print("Request failed:", e)
    except Exception:
        print('xception:', e)


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
            "full_photo": File_Convert_To_Base64(filename=data['full_photo']),
            "source": File_Convert_To_Base64(filename=data['source'])})

        Send_Sos_Request_To_Server(
            url=alarm_url,
            data=data,
            timeout=timeout
        )


def Create_Video_v1(task: dict, sos_tasks: Queue) -> None:
    video_shape = CONF['VIDEO_SHAPE']
    color = CONF['PERSON_WITH_WEAPON']
    bitrate = CONF['BITRATE']
    fps = CONF['FPS']
    weapon_color = CONF['WEAPON_COLOR']

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


def Create_Video_v2(task: dict, sos_tasks: Queue) -> None:
    global VIDEO_CREATE_COUNTER

    video_shape = CONF['VIDEO_SHAPE']
    color = CONF['PERSON_WITH_WEAPON']
    bitrate = CONF['BITRATE']

    camera: str = task['camera']
    camera_fps: int = task['camera_fps']
    camera_shape: tuple[int, int] = task['camera_shape']
    frames: list[tuple[int, ndarray]] = task['frames']
    people: list[dict[str, Any]] = task['people']

    (
        emit_image_path,
        full_image_path,
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
    suspicious_people_frame_id: dict = people[0]['frame_id']
    while suspicious_people_frame_id <= frame_index:
        info: dict = people.pop(0)
        suspicious_people_frame_id = info['frame_id']

    frame_ids = [frame_index for frame_index, _ in frames]
    people = Create_Interpolate_Person_Coors(frame_ids=frame_ids,
                                             people=people)

    image_saved = False
    suspicious_people = people.pop(0)
    for frame_index, frame_data in frames:
        if frame_index == suspicious_people['frame_id']:
            for person in suspicious_people['data']:
                x1, y1, x2, y2 = Transform_Point(point=person,
                                                 from_shape=camera_shape,
                                                 to_shape=video_shape)

                cv2.rectangle(img=frame_data,
                              pt1=(x1, y1), pt2=(x2, y2),
                              color=color, thickness=2, lineType=2)
                if not image_saved:
                    # Save full image
                    cv2.imwrite(filename=full_image_path, img=frame_data)

                    # Save human image
                    w, h = frame_data.shape[:2][::-1]
                    _x1, _y1, _x2, _y2 = Cut_New_Coor(x1, y1, x2, y2, w, h)
                    __only_human = frame_data[_y1:_y2, _x1:_x2]
                    cv2.imwrite(filename=emit_image_path, img=__only_human)
                    image_saved = True
            if len(people):
                suspicious_people = people.pop(0)
        videoWriter.write(image=frame_data)

    if videoWriter.isOpened():
        videoWriter.release()

    print('# $ _____________________________________________________________________________ $ #')
    print('# $ ______________  V I D E O   C R E A T I N G  I S  S T A R T E D _____________ $ #')
    Compress_Video(
        input_video_path=real_video_path,
        output_video_path=compressed_video_path,
        bitrate=bitrate)
    print('# $ _________________________  V I D E O   C R E A T E D ________________________ $ #')
    print('# $ _____________________________________________________________________________ $ #')

    print('Alarm send task is added...')
    sos_tasks.put({
        'camera_ip': camera,
        'the_date': emit_time,
        'event_photo': emit_image_path,
        'full_photo': full_image_path,
        'source': compressed_video_path,
    })
    VIDEO_CREATE_COUNTER += 1


def Create_Video_v3(task: dict, sos_tasks: Queue) -> None:
    global VIDEO_CREATE_COUNTER

    video_shape = CONF['VIDEO_SHAPE']
    color = CONF['PERSON_WITH_WEAPON']
    bitrate = CONF['BITRATE']

    camera: str = task['camera']
    camera_fps: int = task['camera_fps']
    camera_shape: tuple[int, int] = task['camera_shape']
    frames: list[tuple[int, ndarray]] = task['frames']
    people: list[dict[str, Any]] = task['people']

    (
        emit_image_path,
        full_image_path,
        compressed_video_path,
        real_video_path,
        emit_time
    ) = Create_Output_Filenames(camera=camera)

    people = Interpolate_Person_Coordinates_V1(coors=people)
    # people = interpolate_person_coordinates(coors=people)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoWriter = cv2.VideoWriter(
        filename=real_video_path,
        fourcc=fourcc,
        fps=camera_fps,
        frameSize=video_shape)

    print('Person ID:', task['person'], 'People Len:',
          len(people), 'Camera FPS:', camera_fps, 'Frames Len:', len(frames))
    frame_index, _ = frames[0]
    suspicious_people_frame_id: dict = people[0]['frame_id']
    while suspicious_people_frame_id <= frame_index:
        info: dict = people.pop(0)
        suspicious_people_frame_id = info['frame_id']

    image_saved = False
    suspicious_people = people.pop(0)
    for frame_index, frame_data in frames:
        if frame_index == suspicious_people['frame_id']:
            point = (suspicious_people['x1'], suspicious_people['y1'],
                     suspicious_people['x2'], suspicious_people['y2'])
            x1, y1, x2, y2 = Transform_Point(point=point,
                                             from_shape=camera_shape,
                                             to_shape=video_shape)
            cv2.rectangle(img=frame_data,
                          pt1=(x1, y1), pt2=(x2, y2),
                          color=color, thickness=2, lineType=2)
            if not image_saved:
                # Save full image
                cv2.imwrite(filename=full_image_path, img=frame_data)

                # Save human image
                w, h = frame_data.shape[:2][::-1]
                _x1, _y1, _x2, _y2 = Cut_New_Coor(x1, y1, x2, y2, w, h)
                __only_human = frame_data[_y1:_y2, _x1:_x2]
                cv2.imwrite(filename=emit_image_path, img=__only_human)
                image_saved = True
            if len(people):
                suspicious_people = people.pop(0)
        videoWriter.write(image=frame_data)

    if videoWriter.isOpened():
        videoWriter.release()

    print('Video is done!')
    print('# $ _____________________________________________________________________________ $ #')
    print('# $ ______________  V I D E O   C R E A T I N G  I S  S T A R T E D _____________ $ #')
    Compress_Video(
        input_video_path=real_video_path,
        output_video_path=compressed_video_path,
        bitrate=bitrate)
    print('# $ _________________________  V I D E O   C R E A T E D ________________________ $ #')
    print('# $ _____________________________________________________________________________ $ #')

    print('Alarm send task is added...')
    sos_tasks.put({
        'camera_ip': camera,
        'the_date': emit_time,
        'event_photo': emit_image_path,
        'full_photo': full_image_path,
        'source': compressed_video_path,
    })
    VIDEO_CREATE_COUNTER += 1


def Create_Video_v4(task: dict, sos_tasks: Queue) -> None:
    global VIDEO_CREATE_COUNTER

    video_shape = CONF['VIDEO_SHAPE']
    color = CONF['PERSON_WITH_WEAPON']
    bitrate = CONF['BITRATE']

    camera: str = task['camera']
    person: int = task['person']
    camera_fps: int = task['camera_fps']
    camera_shape: tuple[int, int] = task['camera_shape']
    frames: list[tuple[int, ndarray]] = task['frames']
    suspicious: set[int] = task['suspicious']
    people: dict[int, dict] = task['people']

    (
        emit_image_path,
        full_image_path,
        compressed_video_path,
        real_video_path,
        emit_time
    ) = Create_Output_Filenames(camera=camera)

    frame_ids = [frame_index for frame_index, _ in frames]
    people: dict[int, dict] = Interpolate_Person_Coordinates_V2(people=people,
                                                                frame_ids=frame_ids,
                                                                suspicious=suspicious)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoWriter = cv2.VideoWriter(
        filename=real_video_path,
        fourcc=fourcc,
        fps=camera_fps,
        frameSize=video_shape)

    image_saved = False
    for frame_index, frame_data in frames:
        if frame_index in people:
            for person_id in people[frame_index].keys():
                point = (people[frame_index][person_id]['x1'], people[frame_index][person_id]['y1'],
                         people[frame_index][person_id]['x2'], people[frame_index][person_id]['y2'])
                x1, y1, x2, y2 = Transform_Point(point=point,
                                                 from_shape=camera_shape,
                                                 to_shape=video_shape)
                cv2.rectangle(img=frame_data,
                              pt1=(x1, y1), pt2=(x2, y2),
                              color=color, thickness=2, lineType=2)
                if not image_saved and person_id == person:
                    # Save full image
                    cv2.imwrite(filename=full_image_path, img=frame_data)

                    # Save human image
                    w, h = frame_data.shape[:2][::-1]
                    _x1, _y1, _x2, _y2 = Cut_New_Coor(x1, y1, x2, y2, w, h)
                    __only_human = frame_data[_y1:_y2, _x1:_x2]
                    cv2.imwrite(filename=emit_image_path, img=__only_human)
                    image_saved = True

        videoWriter.write(image=frame_data)

    if videoWriter.isOpened():
        videoWriter.release()

    print('Video is done!')

    print('# $ _____________________________________________________________________________ $ #')
    print('# $ ______________  V I D E O   C R E A T I N G  I S  S T A R T E D _____________ $ #')
    Compress_Video(
        input_video_path=real_video_path,
        output_video_path=compressed_video_path,
        bitrate=bitrate)
    print('# $ _________________________  V I D E O   C R E A T E D ________________________ $ #')
    print('# $ _____________________________________________________________________________ $ #')

    print('Alarm send task is added...')
    sos_tasks.put({
        'camera_ip': camera,
        'the_date': emit_time,
        'event_photo': emit_image_path,
        'full_photo': full_image_path,
        'source': compressed_video_path,
    })
    VIDEO_CREATE_COUNTER += 1


def Video_Creator(video_create_task: Queue, sos_tasks: Queue) -> None:
    global VIDEO_CREATE_COUNTER

    while True:
        if video_create_task.empty():
            time.sleep(0.1)
            continue

        if VIDEO_CREATE_COUNTER == 0:
            time.sleep(1)
            continue

        task: Union[dict, None] = video_create_task.get()
        if task is None:
            break

        VIDEO_CREATE_COUNTER -= 1
        thread = Thread(
            target=Create_Video_v4,
            args=(task, sos_tasks,)
        )
        thread.start()
        print('Thread is opened...')
