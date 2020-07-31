import argparse

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *

import imutils.video
from flask import Flask, render_template, Response, request
import time

#from camera_opencv import Camera
import os
import cv2
from base_camera import BaseCamera
from colorama import Fore, Back, Style, init
init(autoreset=True)

app = Flask(__name__)

CATEGORY = None
INFERENCE_STATUS = True
class CCTV(BaseCamera):
    def __init__(self, source):
        CCTV.set_video_source(source)
        super(CCTV, self).__init__()

    @staticmethod
    def set_video_source(source):
        CCTV.video_source = source

    @staticmethod
    def frames():
        if INFERENCE_STATUS :
            camera = cv2.VideoCapture(CCTV.video_source)
            time.sleep(2)
            if not camera.isOpened():
                raise RuntimeError('Could not start camera.')

                # encode as a jpeg image and return it
            imgsz = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
            out, source, weights, half, view_img, save_txt = opt.output, CCTV.video_source, opt.weights, opt.half, opt.view_img, opt.save_txt
            webcam = source == '0' or source.startswith('rtmp') or source.startswith('http') or source.endswith('.txt')

            
            # Initialize
            device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
            if os.path.exists(out):
                shutil.rmtree(out)  # delete output folder
            os.makedirs(out)  # make new output folder

            # Initialize model
            model = Darknet(opt.cfg, imgsz)

            # Load weights
            attempt_download(weights)
            if weights.endswith('.pt'):  # pytorch format
                model.load_state_dict(torch.load(weights, map_location=device)['model'])
            else:  # darknet format
                load_darknet_weights(model, weights)

            # Second-stage classifier
            classify = False
            if classify:
                modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
                modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
                modelc.to(device).eval()

            # Eval mode
            model.to(device).eval()


            maxLost = 10   # maximum number of object losts counted when the object is being tracked
            tracker = Tracker(maxLost = maxLost)

            # Fuse Conv2d + BatchNorm2d layers
            # model.fuse()

            # Export mode
            if ONNX_EXPORT:
                model.fuse()
                img = torch.zeros((1, 3) + imgsz)  # (1, 3, 320, 192)
                f = opt.weights.replace(opt.weights.split('.')[-1], 'onnx')  # *.onnx filename
                torch.onnx.export(model, img, f, verbose=False, opset_version=11,
                                input_names=['images'], output_names=['classes', 'boxes'])

                # Validate exported model
                import onnx
                model = onnx.load(f)  # Load the ONNX model
                onnx.checker.check_model(model)  # Check that the IR is well formed
                print(onnx.helper.printable_graph(model.graph))  # Print a human readable representation of the graph
                return

            # Set Dataloader
            vid_path, vid_writer = None, None
            if webcam:
                view_img = True
                torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
                dataset = LoadStreams(source, img_size=imgsz)

            else:
                save_img = True
                dataset = LoadImages(source, img_size=imgsz)

            # Get names and colors
            names = load_classes(opt.names)
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

            while INFERENCE_STATUS:
                print(Fore.GREEN + 'Starting inference .....')
                # read current frame

                # Run inference
                img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
                _ = model(img.half() if half else img.float()) if device.type != 'cpu' else None  # run once
                for path, img, im0s, vid_cap in dataset:
                    if INFERENCE_STATUS:
                        t0 = time.time()
                        img = torch.from_numpy(img).to(device)
                        img = img.half() if half else img.float()  # uint8 to fp16/32
                        img /= 255.0  # 0 - 255 to 0.0 - 1.0
                        if img.ndimension() == 3:
                            img = img.unsqueeze(0)

                        # Inference
                        t1 = torch_utils.time_synchronized()
                        pred = model(img, augment=opt.augment)[0]
                        t2 = torch_utils.time_synchronized()

                        # to float
                        if half:
                            pred = pred.float()

                        # Apply NMS
                        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                                multi_label=False, classes=opt.classes, agnostic=opt.agnostic_nms)

                        # Apply Classifier
                        if classify:
                            pred = apply_classifier(pred, modelc, img, im0s)
                        # Process detections
                        for i, det in enumerate(pred):  # detections for image i
                            if webcam:  # batch_size >= 1
                                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                            else:
                                p, s, im0 = path, '', im0s

                            save_path = str(Path(out) / Path(p).name)
                            s += '%gx%g ' % img.shape[2:]  # print string
                            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
                            detections_bbox = []
                            if det is not None and len(det):
                                # Rescale boxes from imgsz to im0 size
                                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                                # Print results
                                for c in det[:, -1].unique():
                                    n = (det[:, -1] == c).sum()  # detections per class
                                    s += '%g %ss, ' % (n, names[int(c)])  # add to string


                                # Write results
                                for *xyxy, conf, cls in det:
                                    if save_txt:  # Write to file
                                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                                        with open(save_path[:save_path.rfind('.')] + '.txt', 'a') as file:
                                            file.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                                    if view_img:  # Add bbox to image
                                        label = '%s' % (names[int(cls)])
                                        if names[int(cls)] in CATEGORY:   # Modify category what you want.
                                            tracker_anno = plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])
                                            detections_bbox.append(tracker_anno)


                            objects = tracker.update(detections_bbox)

                            #plot_tacker_id(objects, im0)
                            plot_speed_detection(objects, im0, t0)

                            #im01 = cv2.resize(im0, (500, 320), interpolation=cv2.INTER_CUBIC)

                            yield cv2.imencode('.jpg', im0)[1].tobytes()

                



# class Camera_video(BaseCamera):
#     def __init__(self, source):
#         Camera_video.set_video_source(source)
#         super(Camera_video, self).__init__()

#     @staticmethod
#     def set_video_source(source):
#         Camera_video.video_source = source

#     @staticmethod
#     def frames():
#         camera = cv2.VideoCapture(Camera_video.video_source)

#         while camera.isOpened():
#             # read current frame
#             _, img = camera.read()

#             # encode as a jpeg image and return it
#             yield cv2.imencode('.jpg', img)[1].tobytes()


@app.route('/cctv', methods=['GET', 'POST'])
def index():
    """Video streaming home page."""
    global INFERENCE_STATUS
    parm = request.args
    category = parm.get('category')
    source = parm.get('source')
    if request.method == 'POST':
        btn = request.form.get('button')
        if btn == 'start':
            INFERENCE_STATUS = True 
        elif btn == 'stop':
            INFERENCE_STATUS = False


    return render_template('index.html', source=source, category=category)


def gen(camera):
    """Video streaming generator function."""
    while INFERENCE_STATUS:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    global CATEGORY

    parm = request.args
    category = parm.get('category')
    source = parm.get('source')

    if category == 'person':
        CATEGORY = ['person']
    elif category == 'vehicle':
        CATEGORY = ['car']
    
    return Response(gen(CCTV(source)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='weights/yolov3.pt', help='weights path')
    parser.add_argument('--source', type=str, default='rtmp://3.115.116.226/demo/live', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    opt.names = check_file(opt.names)  # check file

    app.run(host='0.0.0.0', port=8080, thread=True)


    # with torch.no_grad():
    #     detect()  s
