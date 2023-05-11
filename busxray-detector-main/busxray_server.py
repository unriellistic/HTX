import aiofiles, time
from pathlib import Path
from tqdm import tqdm
import cv2, orjson
from sanic import Sanic
from sanic.exceptions import SanicException
from sanic.log import logger
from sanic.response import json
from sanic.worker.loader import AppLoader

from tridentnet_predictor import TridentNetPredictor
import inference as inf

# change this if you want to use a different model
# predictor should be callable, take openCV-format image as input, and output JSON-compatible predictions
predictor = TridentNetPredictor(config_file="models/tridentnet_fast_R_50_C4_3x.yaml",
    opts=["MODEL.WEIGHTS", "models/model_final_e1027c.pkl"]
)




# save the prediction to json file (not needed as this is done on client side)
# output_path = Path(app.config.OUTPUT_FOLDER) / Path(img.name).with_suffix(".json")
# with open(output_path, "wb") as f:
#     f.write(orjson.dumps(predictions, option=orjson.OPT_INDENT_2))
logger.info(f"...done. ({(time.perf_counter() - start_time):.3f} s)")

print(json(non_overlapping_boxes))


