import os, requests
from typing import Union
from pathlib import Path

import orjson
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileMovedEvent

class BusxrayEventHandler(FileSystemEventHandler):
    '''
    Event handler for the busxray system.
    Performs predictions whenever a file is created, moved or modified in the input folder.
    '''
    def __init__(self, target_url: str, output_folder: Union[str, bytes, os.PathLike]):
        self.target_url = target_url
        self.output_folder = output_folder

    def on_any_event(self, event):
        """
        Reacts to events in the folder being watched.
        Doesn't do anything if the event wasn't a file being created, modified or moved, or if the file isn't an image.
        If an image file was created, modified or moved, runs predictions, and dumps the output to the output folder.
        """
        # file creation triggers both FileCreatedEvent and FileModifiedEvent, so don't handle it to avoid duplicate requests
        if type(event) in (FileModifiedEvent, FileMovedEvent):
            file_path = Path(event.src_path)
            if file_path.suffix not in ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'): # not an image, so ignore
                return
            
            # send file to the server
            print(f"Sending file {file_path.name} to {self.target_url}.")
            with open(file_path, "rb") as f:
                response = requests.post(self.target_url, files={"img": (file_path.name, f)})
            print(response.status_code)

            if response.status_code == 200:
                # save the predictions to file
                output_path = Path(self.output_folder) / file_path.with_suffix(".json").name
                with open(output_path, "wb") as f:
                    f.write(orjson.dumps(response.json(), option=orjson.OPT_INDENT_2))

class BusxrayObserver(Observer):
    '''
    Observer class for the busxray system.

    BusxrayObserver.start() will start the listening process.

    The model argument must be a function or a class that implements __call__.
    It takes a PIL.Image as input and outputs predictions in COCO format.
    '''
    def __init__(self, input_folder: Union[str, bytes, os.PathLike], output_folder: Union[str, bytes, os.PathLike], target_url: str):
        super().__init__()
        self.input_folder = input_folder
        self.handler = BusxrayEventHandler(target_url, output_folder)
        self.schedule(self.handler, self.input_folder)