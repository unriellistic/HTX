# busxray-detector

A pipeline to continuously read images from a folder and run inferences (for object detection, etc.)

## Usage
Model weights can be downloaded from [detectron2's TridentNet page](https://github.com/facebookresearch/detectron2/tree/main/projects/TridentNet).

```
$ pip install -r requirements.txt
$ sanic busxray_server (for the server)
$ python busxray_client.py [--source /path/to/folder] [--url TARGET_URL] (for the client)
```

Server configuration (input and output folders) can be found in sanic_config.py. Client configuration (folder to watch and target URL) can be set via command line arguments.

Required files for client: busxray_client.py, busxray_observer.py

Required files for server: busxray_server.py, sanic_config.py, [tridentnet_predictor.py, tridentnet folder OR any other model that you use]

note: `Ikomia-dev/detectron2@v0.6-win10` is used instead of the official detectron2 repository, due to a bug that causes detectron2 installation to fail on Windows.


## Customizing the predictor
The predictor can be any `Callable`. It must take an OpenCV-format image as input, and output COCO format (JSON-compatible) predictions.
