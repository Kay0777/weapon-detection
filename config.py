from configparser import ConfigParser
from argparse import ArgumentParser
from typing import Any
import ast

from os.path import exists, join, dirname, isfile
from os import makedirs


class SettingsBase(type):
    def __call__(cls, *args: tuple, **kwargs: dict) -> Any:
        if not hasattr(cls, 'instance'):
            cls.instance = super().__call__(*args, **kwargs)
        return cls.instance


class Settings(metaclass=SettingsBase):
    def __init__(self, default_config_file: str = 'config.ini'):
        if not exists(default_config_file) or not isfile(default_config_file):
            raise FileNotFoundError(
                f'Config file: [{default_config_file}] file is not found')

        if isfile(default_config_file):
            default_config_file = join(dirname(__file__), default_config_file)

        parser = ArgumentParser(
            description='Example script with argparse and config file.')

        # Add the -c/--config parameter
        parser.add_argument('-c', '--config',
                            default=default_config_file,
                            help='Specify the config file path')

        args = parser.parse_args()

        config = ConfigParser()
        config.read(args.config)

        # Access values from the configuration file
        self.__settings = {
            # Getting Main Path
            "PATH": dirname(__file__),

            "CAMERA_IPS": ast.literal_eval(self.clean(config.get('Main Part', 'CAMERA_IPS'))),
            "LOGIN_PASSWORDS": list(zip(
                ast.literal_eval(
                    self.clean(config.get('Main Part', 'CAMERA_LOGINS'))),
                ast.literal_eval(
                    self.clean(config.get('Main Part', 'CAMERA_PASSWORDS')))
            )),

            # Getting settings from Main Part
            "ALARM_INTERVAL": int(self.clean(config.get('Main Part', 'ALARM_INTERVAL'))),
            "LOGGING_FOLDER": join(
                dirname(__file__),
                self.clean(config.get('Main Part', 'LOGGING_FOLDER'))),
            "ERROR_FILENAME": join(
                dirname(__file__),
                self.clean(config.get('Main Part', 'LOGGING_FOLDER')),
                self.clean(config.get('Main Part', 'ERROR_FILENAME'))),
            "INFO_FILENAME": join(
                dirname(__file__),
                self.clean(config.get('Main Part', 'LOGGING_FOLDER')),
                self.clean(config.get('Main Part', 'INFO_FILENAME'))),

            "PER_PROCESS_COUNT": int(self.clean(config.get('Main Part', 'PER_PROCESS_COUNT'))),
            "SCALE": int(self.clean(config.get('Main Part', 'SCALE'))),
            "MAX_QUEUE_SIZE": int(self.clean(config.get('Main Part', 'MAX_QUEUE_SIZE'))),
            "DETECTED_FRAMES_COUNT": int(self.clean(config.get('Main Part', 'DETECTED_FRAMES_COUNT'))),
            "WEAPON_IMAGES_SAVE_DIR": join(dirname(__file__), self.clean(config.get('Main Part', 'WEAPON_IMAGES_SAVE_DIR'))),
            "CLASSNAMES": [clss.strip() for clss in self.clean(config.get('Main Part', 'CLASSNAMES'))[1:-1].split(',')],

            # Getting settings from Alarm Part
            "ALARM_URL": '{}{}'.format(
                self.clean(config.get('Alarm Part', 'BASE_URL')),
                self.clean(config.get('Alarm Part', 'ENDPOINT'))),
            "TIMEOUT": int(self.clean(config.get('Alarm Part', 'TIMEOUT'))),

            # Getting settings from Video Part
            "FPS": int(self.clean(config.get('Video Part', 'FPS'))),
            "VIDEO_SHAPE": ast.literal_eval(self.clean(config.get('Video Part', 'VIDEO_SHAPE'))),
            "BITRATE": self.clean(config.get('Video Part', 'BITRATE')),
            "VIDEO_LENGTH": int(self.clean(config.get('Video Part', 'VIDEO_LENGTH'))),
            "WAIT_SECONDS": int(self.clean(config.get('Video Part', 'WAIT_SECONDS'))),
            "WEAPON_COLOR": ast.literal_eval(self.clean(config.get('Video Part', 'WEAPON_COLOR'))),
            "PERSON_WITH_WEAPON": ast.literal_eval(self.clean(config.get('Video Part', 'PERSON_WITH_WEAPON'))),
            "SUSPICIOUS_PERSON": ast.literal_eval(self.clean(config.get('Video Part', 'SUSPICIOUS_PERSON'))),

            # Getting settings from Model Part
            "WEAPON_YOLO_MODEL": join(
                self.clean(
                    config.get('Model Part', 'MODELS_FOLDER')),
                self.clean(
                    config.get('Model Part', 'WEAPON_YOLO_MODEL'))
            ),
            "WEAPON_CLASSIFY_TORCH_MODEL": join(
                self.clean(
                    config.get('Model Part', 'MODELS_FOLDER')),
                self.clean(
                    config.get('Model Part', 'WEAPON_CLASSIFY_TORCH_MODEL'))
            ),
            "WEAPON_TASK": self.clean(config.get('Model Part', 'WEAPON_TASK')),

            # Getting settings from Device Part
            "DEVICE": self.clean(config.get('Device Part', 'DEVICE_TYPE')),
        }

    def clean(self, value: str) -> str:
        return value.split('#')[0].strip()

    def __str__(self) -> str:
        return str(self.__settings)

    def __repr__(self) -> str:
        return str(self.__settings)

    def __getitem__(self, __name: str) -> Any:
        return self.__settings.get(__name, None)

    def items(self) -> list[tuple[str, Any]]:
        return self.__settings.items()

    def keys(self) -> list[str]:
        return self.__settings.keys()


def Create_Logger_Files(log_folder_path: str, error_file: str, info_file: str) -> None:
    makedirs(log_folder_path, exist_ok=True)

    if not exists(error_file):
        open(file=error_file, mode='x')

    if not exists(info_file):
        open(file=info_file, mode='x')


CONF = Settings()
Create_Logger_Files(
    log_folder_path=CONF['LOGGING_FOLDER'],
    error_file=CONF['ERROR_FILENAME'],
    info_file=CONF['INFO_FILENAME']
)

if __name__ == '__main__':
    for key, item in CONF.items():
        print(key, item, type(item))
