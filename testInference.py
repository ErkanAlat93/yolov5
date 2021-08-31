import torch
import torch.backends.cudnn as cudnn
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
import matplotlib.pyplot as plt
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_sync

conf_thres=0.25  # confidence threshold
iou_thres=0.45  # NMS IOU threshold
max_det = 100    # maximum detection per image
agnostic_nms=False  # class-agnostic NMS
classes=0  # filter by class: --class 0, or --class 0 2 3
augment = False
names = ['stairs', 'handrails']
save_txt = True
save_img = True
save_conf = True
hide_conf = False
txt_path = './result'
save_dir = './'
save_crop = True
hide_labels = False
line_thickness = 3
model_path = 'best.pt'


set_logging()
device = select_device('cpu')
model = attempt_load(model_path, map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride

imgsz = check_img_size(640, s=stride)  # check image size

dataset = LoadImages('./data/images/frame0323.jpg', img_size=imgsz, stride=stride, auto=True)
bs = 1


# Getting Image Tensor
for path, img, im0s, vid_cap in dataset:
    img = torch.from_numpy(img).to(device)
    # Normalize Image
    img = img.float()
    img = img / 255.0
    # Expand Dimension --> ([1, 3, 640, 480])
    img = img[None]
    ims = im0s


pred = model(img, augment=augment)[0]
pred = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres, classes=classes, agnostic=agnostic_nms, max_det=max_det)


for i, det in enumerate(pred):
    gn = torch.tensor(ims.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    s = ''
    s += '%gx%g ' % img.shape[2:]  # print string
    if len(det):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], ims.shape).round()

        # Print results
        for c in det[:, -1].unique():
            n = (det[:, -1] == c).sum()  # detections per class
            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
            # print(n, s)
        for *xyxy, conf, cls in reversed(det):
            if save_txt:  # Write to file
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                print(xywh, cls.item(), conf.item())
                with open(txt_path + '.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

            if save_img:  # Add bbox to image
                c = int(cls)  # integer class
                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                ims = plot_one_box(xyxy, ims, label=label, color=colors(c, True), line_width=line_thickness)

                if save_crop:
                    save_one_box(xyxy, ims, file='./test.jpg', BGR=True)