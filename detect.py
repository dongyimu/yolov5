import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from utils.augmentations import letterbox

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator

from models.common import DetectMultiBackend
from utils.general import (
    cv2,
    non_max_suppression,
    scale_boxes,
    xyxy2xywh,
)
from utils.torch_utils import smart_inference_mode


@smart_inference_mode()
def run():
    # Load model
    device = torch.device('cuda:0')
    model = DetectMultiBackend(weights='./weights/yolov5n.pt', device=device, dnn=False, data=False, fp16=True)

    # 读取图片
    im = cv2.imread('data/images/csgo2.jpg')
    im0 = im
    # 处理图片
    im = letterbox(im, (640, 640), stride=32, auto=True)[0]
    im = im.transpose((2, 0, 1))[::-1]
    im = np.ascontiguousarray(im)
    im = torch.from_numpy(im).to(model.device)
    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    # 推理
    start = time.time()
    pred = model(im, augment=False, visualize=False)
    # 非极大值抑制
    pred = non_max_suppression(pred, conf_thres=0.6, iou_thres=0.45, classes=None, max_det=1000)
    end = time.time()
    print(f"推理所需时间{end - start}s")

    # Process predictions
    for i, det in enumerate(pred):  # per image
        annotator = Annotator(im0, line_width=1)
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()  # normalized xywh
                line = cls, *xywh, conf  # label format
                annotator.box_label(xyxy, label=str(int(cls)), color=(34, 139, 34), txt_color=(0, 191, 255))
                print(xywh, line)
        im0 = annotator.result()
        cv2.imshow("window", im0)
        cv2.waitKey(0)


if __name__ == "__main__":
    run()
