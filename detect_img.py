from utils import *
import argparse
import cv2
from dataset import Data_Name_Color
from yolov5s import My_YOLO as my_yolov5s
from yolov5l import My_YOLO as my_yolov5l
from yolov5m import My_YOLO as my_yolov5m
from yolov5x import My_YOLO as my_yolov5x
import numpy as np

def resize_image(srcimg, image_size, keep_ratio=False):
    top, left, newh, neww = 0, 0, image_size, image_size
    if keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
        hw_scale = srcimg.shape[0] / srcimg.shape[1]
        if hw_scale > 1:
            newh, neww = image_size, int(image_size / hw_scale)
            img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
            left = int((image_size - neww) * 0.5)
            img = cv2.copyMakeBorder(img, 0, 0, left, image_size - neww - left, cv2.BORDER_CONSTANT, value=0)  # add border
        else:
            newh, neww = int(image_size * hw_scale), image_size
            img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
            top = int((image_size - newh) * 0.5)
            img = cv2.copyMakeBorder(img, top, image_size - newh - top, 0, 0, cv2.BORDER_CONSTANT, value=0)
    else:
        img = cv2.resize(srcimg, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return img, newh, neww, top, left

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, default='2008_003261.jpg', help="image path")
    parser.add_argument('--labels_txt', default='pascal_voc.names', help='labels.txt')
    parser.add_argument('--imgsize', type=int, default=640, help='image size')
    parser.add_argument('--input_norm', action='store_true', help='Input Normliaze')
    parser.add_argument('--pth', type=str, default='pascal_voc.pth', help='train pth path')
    parser.add_argument('--net_type', default='yolov5s', choices=['yolov5s', 'yolov5l', 'yolov5m', 'yolov5x'])
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument('--keep_ratio', action='store_true', help='resize image keep ratio')
    args = parser.parse_args()
    print(args)

    data_color = Data_Name_Color(args.labels_txt)
    # Set up model
    anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
    if args.net_type == 'yolov5s':
        net = my_yolov5s(data_color.num_classes, anchors=anchors, training=False)
    elif args.net_type == 'yolov5l':
        net = my_yolov5l(data_color.num_classes, anchors=anchors, training=False)
    elif args.net_type == 'yolov5m':
        net = my_yolov5m(data_color.num_classes, anchors=anchors, training=False)
    else:
        net = my_yolov5x(data_color.num_classes, anchors=anchors, training=False)
    net.to(device)
    net.load_state_dict(torch.load(args.pth, map_location=device))
    net.eval()

    srcimg = cv2.imread(args.img_path)
    print(args.img_path)
    # img = cv2.resize(srcimg, (args.imgsize, args.imgsize), interpolation=cv2.INTER_AREA)
    img, newh, neww, top, left = resize_image(srcimg, args.imgsize, keep_ratio=args.keep_ratio)
    img = np.ascontiguousarray(img, dtype=np.float32)
    if args.input_norm:
        img /= 255
    input_img = torch.from_numpy(img).to(device)
    input_img = input_img.permute(2,0,1)
    input_img = input_img.unsqueeze(0)

    detections = net(input_img)[0]
    detections = non_max_suppression(detections.detach().cpu(), conf_thres=args.conf_thres, iou_thres=args.iou_thres, classes=data_color.num_classes)[0]
    if detections is not None:
        ratioh,ratiow = srcimg.shape[0]/newh,srcimg.shape[1]/neww
        drawimg = srcimg.copy()
        for det in detections:
            # if det[4].item() < 0.8:
            #     continue
            label = data_color.class_names[int(det[5])]
            color = data_color.colors[int(det[5])]
            # xmin,ymin,xmax,ymax = int(det[0]*ratiow), int(det[1]*ratioh), int(det[2]*ratiow), int(det[3]*ratioh)
            xmin, ymin, xmax, ymax = max(int((det[0]-left) * ratiow), 0), max(int((det[1]-top) * ratioh), 0), min(
                int((det[2]-left) * ratiow), srcimg.shape[1]), min(int((det[3]-top) * ratioh), srcimg.shape[0])
            cv2.rectangle(drawimg, (xmin, ymin), (xmax, ymax), color, thickness=4)
            cv2.putText(drawimg, label, (xmin, ymin-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), thickness=2)

        # cv2.imwrite('result.jpg', drawimg)
        cv2.namedWindow('detect', cv2.WINDOW_NORMAL)
        cv2.imshow('detect', drawimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()