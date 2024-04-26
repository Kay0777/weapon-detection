from src import StreamProcessor1
from src import StreamProcessor2
from src.video_creater_and_sos import (
    Video_Creator_v1,
    Video_Creator_v2,
    Sos_Sender
)
from src.utils import (
    Write_Error_MSG,
    Write_Info_MSG
)
import os
import time
from queue import Queue
from threading import Thread
from multiprocessing import (
    Process,
    Manager,
    cpu_count
)

from config import CONF


def Run_Stream_Process(camera: str, login_passwords: list[tuple[str, str]], video_create_task: Queue, errors_and_info_handle_task: Queue) -> None:
    stream_process = StreamProcessor2(
        camera=camera,
        login_passwords=login_passwords,
        video_create_task=video_create_task,
        errors_and_info_handle_task=errors_and_info_handle_task)
    stream_process.start()


def Process_Cameras(cameras: tuple[str, list[str]], login_passwords: list[tuple[str, str]], video_create_task: Queue, errors_and_info_handle_task: Queue) -> None:
    for camera in cameras:
        thread = Thread(
            target=Run_Stream_Process,
            args=(camera, login_passwords, video_create_task, errors_and_info_handle_task,))
        thread.start()


def Handle_Errors_And_Infos(errors_and_info_handle_task: Queue) -> None:
    while True:
        # time.sleep(30)  # 1 minute
        errors, infos = [], []
        while not errors_and_info_handle_task.empty():
            _type, _msg = errors_and_info_handle_task.get()
            if _type == 'error':
                errors.append(_msg)
            else:
                infos.append(_msg)

        if not errors:
            Write_Error_MSG(msg=errors)

        if not infos:
            Write_Info_MSG(msg=infos)


def Video_Creata_And_Send_Sos(video_tasks: Queue, errors_and_info_handle_task: Queue) -> None:
    sos_tasks: Queue = Queue()
    # Craete Task threads
    create_video_thread = Thread(
        # target=Video_Creator_v1,
        target=Video_Creator_v2,
        args=(video_tasks, sos_tasks,))
    sos_send_thread = Thread(
        target=Sos_Sender,
        args=(sos_tasks,))

    write_messages = Thread(
        target=Handle_Errors_And_Infos,
        args=(errors_and_info_handle_task, )
    )

    # Run Task threads
    create_video_thread.start()
    sos_send_thread.start()
    write_messages.start()


def main1():
    CPU_N = cpu_count()
    PER_PROCESS_COUNT: int = CONF['PER_PROCESS_COUNT']

    cameras = [('192.168.4.{}'.format(i), ['smartbase404', 'parol12345'])
               for i in range(1, 37)]
    new_carmeras = [('10.144.132.194', ['parol12345']), ('10.144.132.198', [
        'parol12345']), ('10.144.132.195', ['123456'])]
    cameras.extend(new_carmeras)

    cameras = [('192.168.4.35', ['smartbase404', 'parol12345'])]

    cameras_per_process = len(cameras) // PER_PROCESS_COUNT

    with Manager() as manager:
        video_create_task = manager.Queue()
        processes: list[Process] = []

        for i in range(PER_PROCESS_COUNT):
            start_index = i * cameras_per_process
            end_index = start_index + cameras_per_process
            camera_subset = cameras[start_index:end_index]

            print(camera_subset)

            # Create and start a Process
            process = Process(target=Process_Cameras,
                              args=(camera_subset, video_create_task, ))
            process.start()

            processes.append(process)

            # Optionally set CPU affinity for each process to limit the CPUs they can run on
            if hasattr(os, 'sched_setaffinity'):
                cpu_subset = range(i * (CPU_N // PER_PROCESS_COUNT),
                                   (i + 1) * (CPU_N // PER_PROCESS_COUNT))
                os.sched_setaffinity(process.pid, cpu_subset)

        video_creater_and_sos_sender_process = Process(target=Video_Creata_And_Send_Sos,
                                                       args=(video_create_task, ))
        video_creater_and_sos_sender_process.start()
        processes.append(video_creater_and_sos_sender_process)

        # Join all run Processes
        for process in processes:
            process.join()


def main2():
    CPU_N = cpu_count()
    PER_PROCESS_COUNT: int = CONF['PER_PROCESS_COUNT']

    cameras = CONF['CAMERA_IPS']
    login_passwords = CONF['LOGIN_PASSWORDS']

    with Manager() as manager:
        video_create_task = manager.Queue()
        errors_and_info_handle_task = manager.Queue()

        processes: list[Process] = []

        for i in range(PER_PROCESS_COUNT):
            start, end = i * PER_PROCESS_COUNT, (i + 1) * PER_PROCESS_COUNT
            camera_subset = cameras[start:end]
            if not camera_subset:
                continue

            # Create and start a Process
            process = Process(
                target=Process_Cameras,
                args=(camera_subset, login_passwords, video_create_task, errors_and_info_handle_task, ))
            process.start()

            processes.append(process)

            # Optionally set CPU affinity for each process to limit the CPUs they can run on
            if hasattr(os, 'sched_setaffinity'):
                cpu_subset = range(i * (CPU_N // PER_PROCESS_COUNT),
                                   (i + 1) * (CPU_N // PER_PROCESS_COUNT))
                os.sched_setaffinity(process.pid, cpu_subset)

        video_creater_and_sos_sender_process = Process(target=Video_Creata_And_Send_Sos,
                                                       args=(video_create_task, errors_and_info_handle_task, ))
        video_creater_and_sos_sender_process.start()
        processes.append(video_creater_and_sos_sender_process)

        # Join all run Processes
        for process in processes:
            process.join()


if __name__ == "__main__":
    # main1()
    main2()
