import os
import sys
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from bottle import route, run, request, post
from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, time_sync


weights=ROOT / 'greyBest(edge Included).pt'  # model.pt path(s)
source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
imgsz=640  # inference size (pixels)
conf_thres=0.6  # confidence threshold
iou_thres=0.45  # NMS IOU threshold
max_det=1000  # maximum detections per image
device=''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
save_txt=True  # save results to *.txt
classes=None  # filter by class: --class 0, or --class 0 2 3
agnostic_nms=False  # class-agnostic NMS
augment=False  # augmented inference
update=False  # update all models
project=ROOT / 'runs/detect'  # save results to project/name
name='exp'  # save results to project/name
exist_ok=False  # existing project/name ok, do not increment
half=False  # use FP16 half-precision inference
dnn=False  # use OpenCV DNN for ONNX inference

check_requirements(exclude=('tensorboard', 'thop'))
# Directories
save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

# Load model
device = select_device(device)
model = DetectMultiBackend(weights, device=device, dnn=dnn)
stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
imgsz = check_img_size(imgsz, s=stride)  # check image size

# Half
half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
if pt:
    model.model.half() if half else model.model.float()


@torch.no_grad()
@post('/detect')
def stand_by():
    # Dataloader
    source = request.json['directory']
    returnFile = {}
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)

    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        pred = model(im, augment=augment, visualize=False)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            chan, w, h = im0.shape
            p = Path(p)  # to Path
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                results = []

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    center_trace = int(w*xywh[0]-74)
                    x_offset = int(w*xywh[2]/2)
                    y_offset = int(chan*xywh[3]/2)
                    center_ns = int(chan*xywh[1]-74)  #*.0134  multiply by .0134 to get ns value
                    hpb = {}
                    hpb["center"] = [center_trace,center_ns]
                    hpb["left_bottom"] = [center_trace-x_offset,center_ns+y_offset]
                    hpb["right_bottom"] = [center_trace+x_offset,center_ns+y_offset]
                    hpb["confidence"] = float(conf)
                    results.append(hpb)
                
                results.sort(key=lambda x: x["center"][0])
                LOGGER.info('Results are:')
                LOGGER.info(results)

                returnFile[txt_path.split('/')[-1]] = results

            # Print time (inference-only)
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

    return(returnFile)

if __name__ == "__main__":
    run(host="localhost", port=8080, debug=True)