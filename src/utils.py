from moviepy.editor import VideoFileClip
import base64
import time
import os

from config import CONF
from typing import Any


class SuspiciousPeople:
    def __init__(self) -> None:
        self.__data: list[dict] = []

    def __iadd__(self, data: Any) -> None:
        self.__data.append(data)
        return self

    def __next__(self) -> Any:
        if self.is_empty():
            return self, None
        return self, self.__data.pop(0)

    def add(self, data: dict) -> None:
        self.__data.append(data)

    def join(self, data: list[dict]) -> None:
        self.__data.extend(data)

    def is_empty(self) -> bool:
        return len(self.__data) == 0

    def __str__(self) -> str:
        return str(self.__data)


def File_Convert_To_Base64(filename: str) -> str:
    with open(file=filename, mode="rb") as file:
        data = base64.b64encode(file.read())
    return data.decode('utf-8')


def Compress_Video(input_video_path: str, output_video_path: str, bitrate: str) -> None:
    clip = VideoFileClip(filename=input_video_path)
    clip.write_videofile(filename=output_video_path, bitrate=bitrate)
    clip.close()


def Get_Until_This_Frame_ID(alarm_frame_id: int) -> tuple[int, int]:
    fps: int = CONF['FPS']
    seconds_of_waiting: int = CONF['WAIT_SECONDS']
    video_length: int = CONF['VIDEO_LENGTH']

    frame_from = max(1, alarm_frame_id - fps * seconds_of_waiting)
    frame_to = frame_from + fps * video_length

    return frame_from, frame_to


def Create_Output_Filenames(camera: str) -> tuple[str, str, str, str]:
    main_path = os.path.join('videos', camera)
    os.makedirs(name=main_path, exist_ok=True)

    emit_time = time.strftime('%d.%m.%Y %H:%M:%S')
    the_time = time.strftime('%d_%m_%Y__%H_%M_%S')

    emit_image_path = os.path.join(
        main_path, f'weapon__{the_time}__image.jpg')
    compressed_video_path = os.path.join(
        main_path, f'weapon__{the_time}__compressed__video.mp4')
    real_video_path = os.path.join(
        main_path, f'weapon__{the_time}__real__video.mp4')

    return (
        emit_image_path,
        compressed_video_path,
        real_video_path,
        emit_time
    )
