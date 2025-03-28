import argparse
import os
import platform
import sys
from pathlib import Path
import math
import torch
import numpy as np
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from collections import deque
import cv2
import pyautogui

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

def initialize_deepsort():
    # Create the Deep SORT configuration object and load settings from the YAML file
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")

    # Initialize the DeepSort tracker
    deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                        max_dist=cfg_deep.DEEPSORT.MAX_DIST,
                        # min_confidence  parameter sets the minimum tracking confidence required for an object detection to be considered in the tracking process
                        min_confidence=.1,
                        # nms_max_overlap specifies the maximum allowed overlap between bounding boxes during non-maximum suppression (NMS)
                        nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP,
                        # max_iou_distance parameter defines the maximum intersection-over-union (IoU) distance between object detections
                        max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                        # Max_age: If an object's tracking ID is lost (i.e., the object is no longer detected), this parameter determines how many frames the tracker should wait before assigning a new id
                        max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT,
                        # nn_budget: It sets the budget for the nearest-neighbor search.
                        nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                        use_cuda=True
                        )

    return deepsort

deepsort = initialize_deepsort()
data_deque = {}
def classNames():
    cocoClassNames = ["Cell"
                  ]
    return cocoClassNames
className = classNames()

def colorLabels(classid):
    if classid == 0: #person
        color = (85, 45, 255)
    elif classid == 2: #car
        color = (222, 82, 175)
    elif classid == 3: #Motorbike
        color = (0, 204, 255)
    elif classid == 5: #Bus
        color = (0,149,255)
    else:
        color = (200, 100,0)
    return tuple(color)

def draw_boxes(frame, bbox_xyxy, draw_trails, identities=None, categories=None, offset=(0,0)):
    height, width, _ = frame.shape
    for key in list(data_deque):
      if key not in identities:
        data_deque.pop(key)

    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        y1 += offset[0]
        x2 += offset[0]
        y2 += offset[0]
        #Find the center point of the bounding box
        center = int((x1+x2)/2), int((y1+y2)/2)
        cat = int(categories[i]) if categories is not None else 0
        color = colorLabels(cat)
        #color = [255,0,0]#compute_color_labels(cat)
        id = int(identities[i]) if identities is not  None else 0
        # create new buffer for new object
        if id not in data_deque:
          data_deque[id] = deque(maxlen= 64)
        data_deque[id].appendleft(center)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        name = className[cat]
        label = str(id) + ":" + name
        text_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0]
        c2 = x1 + text_size[0], y1 - text_size[1] - 3
        cv2.rectangle(frame, (x1, y1), c2, color, -1)
        cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
        cv2.circle(frame,center, 2, (0,255,0), cv2.FILLED)
        if draw_trails:
              # draw trail
              for i in range(1, len(data_deque[id])):
                  # check if on buffer value is none
                  if data_deque[id][i - 1] is None or data_deque[id][i] is None:
                      continue
                  # generate dynamic thickness of trails
                  thickness = int(np.sqrt(64 / float(i + i)) * 1.5)
                  # draw trails
                  cv2.line(frame, data_deque[id][i - 1], data_deque[id][i], color, thickness)    
    return frame

manual_bounding_boxes = [
    [1160, 111, 1175, 135, 0.12, 0.0],
    [1085, 138, 1100, 162, 0.12, 0.0],
    [1088, 140, 1103, 164, 0.12, 0.0],
    [1091, 142, 1106, 166, 0.12, 0.0],
    [1094, 144, 1109, 168, 0.12, 0.0],
    [1097, 146, 1112, 170, 0.12, 0.0],
    [1100, 148, 1115, 172, 0.12, 0.0],
    [1103, 150, 1118, 174, 0.12, 0.0],
    [1106, 152, 1121, 176, 0.12, 0.0],
    [1109, 154, 1124, 178, 0.12, 0.0],
    [1112, 156, 1127, 180, 0.12, 0.0],
    [1115, 158, 1130, 182, 0.12, 0.0],
    [1118, 160, 1133, 184, 0.12, 0.0],
    [1121, 162, 1136, 186, 0.12, 0.0],
    [1124, 164, 1139, 188, 0.12, 0.0],
    [1127, 166, 1142, 190, 0.12, 0.0],
    [1130, 168, 1145, 192, 0.12, 0.0],
    [1133, 170, 1148, 194, 0.12, 0.0],
    [1136, 172, 1151, 196, 0.12, 0.0],
    [1139, 174, 1154, 198, 0.12, 0.0],
    [1142, 176, 1157, 200, 0.12, 0.0],
    [1145, 178, 1160, 202, 0.12, 0.0],
    [1148, 180, 1163, 204, 0.12, 0.0],
    [1151, 182, 1166, 206, 0.12, 0.0],
    [1154, 184, 1169, 208, 0.12, 0.0],
    [1157, 186, 1172, 210, 0.12, 0.0],
    [1160, 188, 1175, 212, 0.12, 0.0],
    [1163, 190, 1178, 214, 0.12, 0.0],
    [1166, 192, 1181, 216, 0.12, 0.0],
    [1169, 194, 1184, 218, 0.12, 0.0],
]

STATIC_WIDTH = 20
STATIC_HEIGHT = 27

@smart_inference_mode()
def run(
        weights=ROOT / 'best.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.1,  # confidence threshold
        iou_thres=0.5,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=True,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        draw_trails = False,
        ):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs
    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    def crop_frame(im0, patch_size=640, overlap=240):
        h, w, _ = im0.shape
        patches = []
        coords = []

        # Calculate the step size based on overlap
        step_size = patch_size - overlap

        # Ensure we cover the entire height
        for y in range(0, h, step_size):
            for x in range(0, w, step_size):
                # Calculate the end coordinates of the patch
                x_end = min(x + patch_size, w)
                y_end = min(y + patch_size, h)

                # Adjust the starting coordinates if the patch goes out of bounds
                if x_end - x < patch_size and x_end < w:
                    x = x_end - patch_size
                if y_end - y < patch_size and y_end < h:
                    y = y_end - patch_size

                # Extract the patch
                patch = im0[y:y_end, x:x_end]
                patches.append(patch)
                coords.append((x, y))

        return patches, coords

    # Main loop
    out = None
    frame_number = 0  # Initialize the frame counter
    fc = 0
    fake_count = len(manual_bounding_boxes)
    for path, im, im0s, vid_cap, s in dataset:
        im0 = im0s.copy()
        patch_size = 640
        all_detections = []  # To store detections from all patches

        # Initialize VideoWriter on the first frame
        if out is None:
            frame_height, frame_width = im0s.shape[:2]  # Height, Width
            frame_size = (frame_width, frame_height)
            fps = vid_cap.get(cv2.CAP_PROP_FPS) if vid_cap else 30  # Use video capture for FPS
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
            output_video_path = 'output_tracking.mp4'
            out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

        # Step 1: Crop the frame
        patches, coords = crop_frame(im0, patch_size)

        for patch, (x_offset, y_offset) in zip(patches, coords):
            # Resize and preprocess each patch
            resized_patch = cv2.resize(patch, (patch_size, patch_size))
            im = torch.from_numpy(resized_patch).to(model.device)

            # Transpose to match PyTorch model input format: [batch_size, channels, height, width]
            im = im.permute(2, 0, 1).unsqueeze(0)  # Convert [H, W, C] to [1, C, H, W]

            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # Normalize to [0, 1]

            # Now proceed with the rest of your pipeline
            pred = model(im, augment=augment)[0]
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

            # Adjust detection boxes to full-frame coordinates
            for det in pred:
                if det is not None and len(det):  # Check if there are detections
                    # Scale detection boxes
                    scaled_boxes = scale_boxes((patch_size, patch_size), det[:, :4], patch.shape).round()
                    det[:, :4] = scaled_boxes  # Update detection boxes
                    det[:, [0, 2]] += x_offset  # Add x-offset
                    det[:, [1, 3]] += y_offset  # Add y-offset
                    #print(f'The offsets X {x_offset} and Y {y_offset}')

                    # Validate detections
                    valid_detections = []
                    for d in det:
                        x1, y1, x2, y2 = map(int, d[:4])
                        if (x2 - x1) >= 1 and (y2 - y1) >= 1:  # Check for valid bounding box size
                            if x1 < 0 or y1 < 0:
                                print(f"Detection out of bounds skipped: {d}")
                                continue
                            valid_detections.append(d.tolist())

                    all_detections.extend(valid_detections)

        #if frame_number < len(manual_bounding_boxes):
        #    all_detections.append(manual_bounding_boxes[frame_number])

        # Step 3: Combine detections from all patches
        if len(all_detections) == 0:
            print("No detections found in any patches.")
            continue
        #print('THESE ARE ALL THE CONTENTS OF DETECTIONS!', all_detections)
        if all_detections:
            # Pass combined detections to DeepSORT
            xywh_bboxs = []
            confs = []
            oids = []
            for *xyxy, conf, cls in all_detections:
                x1, y1, x2, y2 = map(int, xyxy)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                w, h = x2 - x1, y2 - y1

                #print(f"Detection - x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}, conf: {conf}, cls: {cls}")
                xywh_bboxs.append([cx, cy, w, h])
                confs.append(conf)
                oids.append(int(cls))
            xywhs = torch.tensor(xywh_bboxs)

            # Write frame number once for the current frame
            with open("bounding_boxes.txt", "a") as file:
                file.write(f"Frame: {frame_number}\n")
            frame_number +=1

            for bbox, oid in zip(xywh_bboxs, oids):
                cx, cy, w, h = map(int, bbox)  # Convert bounding box to integers
                x1 = cx - STATIC_WIDTH // 2  # Top-left x
                y1 = cy - STATIC_HEIGHT // 2  # Top-left y
                x2 = cx + STATIC_WIDTH // 2  # Bottom-right x
                y2 = cy + STATIC_HEIGHT // 2  # Bottom-right y

                # Draw bounding box and center coordinates on the image
                cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw a green rectangle for the box
                cv2.putText(im0, f"({cx},{cy})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Append bounding box and frame number information to a text file
                #with open("bounding_boxes.txt", "a") as file:  # Open file in append mode
                    #file.write(f"{cx} {cy} {w} {h}\n")

            # Show the image with current detections
            cv2.imshow("Current Detections", im0)

            # Add a delay and allow for exit
            if cv2.waitKey(1) == ord('q'):  # Press 'q' to quit visualization
                break

            # Update DeepSORT tracking
            if len(xywh_bboxs) == 0:
                outputs = deepsort.update(None, None, None, im0)
            else:
                xywhs = torch.tensor(xywh_bboxs, dtype=torch.float32)
                confss = torch.tensor(confs, dtype=torch.float32)
                outputs = deepsort.update(xywhs, confss, oids, im0)

            # Draw bounding boxes on the original frame
            if len(outputs) > 0:
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -2]
                object_id = outputs[:, -1]
                with open("info.txt", "a") as file:  # Open file in append mode
                    file.write(f"Frame: {frame_number}\n{np.column_stack((identities,bbox_xyxy))}\n")
                draw_boxes(im0, bbox_xyxy, draw_trails, identities, object_id)

        # Write the frame to the output video
        if out is not None:
            out.write(im0)
        # Display or save the results
        if view_img:
            cv2.imshow("Tracking", im0)
            if cv2.waitKey(1) == ord('q'):
                break

    cv2.destroyAllWindows()
    if out is not None:
        out.release()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolo.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.15, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.3, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=80, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--draw-trails', action='store_true', help='do not drawtrails')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt 


def main(opt):
    # check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)