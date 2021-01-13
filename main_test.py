from common import *
from utils import *
from loss_fun import compute_loss
from yolov5s import My_YOLO as my_yolov5s
from yolov5l import My_YOLO as my_yolov5l
from yolov5m import My_YOLO as my_yolov5m
from yolov5x import My_YOLO as my_yolov5x
from dataset import MyDataset_xml, MyDataset_json
from torch.utils.data import DataLoader
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def evaluate(model, dataloader, iou_thres, conf_thres, nms_thres, img_size):
    model.yolo_layers.training = False
    model.eval()
    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    nb, all_loss = len(dataloader), 0
    pbar = tqdm(enumerate(dataloader), desc="test", total=nb)  # progress bar
    with torch.no_grad():
        for batch_i, (imgs, targets) in pbar:
            # srcimg = imgs[0,:,:,:].detach().cpu().numpy().astype(np.uint8)
            imgs = imgs.to(device).permute(0, 3, 1, 2)
            targets = targets.to(device)
            # outputs, loss = model(imgs)
            outputs, pred = model(imgs)
            loss, loss_items = compute_loss(pred, targets, model)
            all_loss += loss.item()

            # Extract labels
            labels += targets[:, 1].tolist()
            # Rescale target
            targets[:, 2:] = xywh2xyxy(targets[:, 2:])
            targets[:, 2:] *= img_size
            outputs = non_max_suppression(to_cpu(outputs), conf_thres=conf_thres, iou_thres=iou_thres)
            sample_metrics += get_batch_statistics(outputs, to_cpu(targets), iou_threshold=iou_thres)
    # Concatenate sample statistics
    # true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    # precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)
    # return precision, recall, AP, f1, ap_class, AP.mean(), all_loss / nb
    if len(sample_metrics)>0:
        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
        precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)
        return precision, recall, AP, f1, ap_class, AP.mean(), all_loss/nb
    else:
        return all_loss/nb

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgroot', default='/home/wangbo/Desktop/data/datasets/VOCdevkit/VOC2012/Annotations', help='image directory path')
    parser.add_argument('--labroot', default='/home/wangbo/Desktop/data/datasets/VOCdevkit/VOC2012/Annotations', help='label directory path')
    parser.add_argument('--test_txt', default='/home/wangbo/Desktop/data/datasets/VOCdevkit/VOC2012/ImageSets/Main/val.txt', help='test.txt')
    parser.add_argument('--labels_txt', default='/home/wangbo/Desktop/data/datasets/VOCdevkit/pascal_voc.names', help='labels.txt')
    parser.add_argument('--imgsize', type=int, default=640, help='image size')
    parser.add_argument('--input_norm', action='store_true', help='Input Normliaze')
    parser.add_argument('--batchsize', type=int, default=1, help='Batch size, mv2:16, mv3:32')
    parser.add_argument('--pth', type=str, default='pascal_voc/best.pth', help='train pth path')
    parser.add_argument('--saveout', type=str, default='', help='save detect image to dir')
    parser.add_argument("--sample_type", default='nearest', choices=['deconv', 'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear', 'area'])
    parser.add_argument('--net_type', default='yolov5s', choices=['yolov5s', 'yolov5l', 'yolov5m', 'yolov5x'])
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    args = parser.parse_args()
    print(args)

    testDataSet = MyDataset_xml(args.imgroot, args.labroot, args.test_txt, args.labels_txt, image_size=args.imgsize,
                                input_norm=args.input_norm)
    # testDataSet = MyDataset_json(args.imgroot, args.labroot, args.test_txt, args.labels_txt, image_size=args.imgsize, input_norm=args.input_norm, train=False)
    print("validation dataset size: {}".format(len(testDataSet)))
    dataloaderTest = DataLoader(testDataSet,
                                batch_size=args.batchsize,
                                shuffle=False,
                                num_workers=0,
                                pin_memory=True,
                                collate_fn=testDataSet.collate_fn)

    anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
    # net = My_YOLO(trainDataSet.num_classes, anchors=anchors)
    if args.net_type == 'yolov5s':
        net = my_yolov5s(testDataSet.num_classes, anchors=anchors, training=True)
    elif args.net_type == 'yolov5l':
        net = my_yolov5l(testDataSet.num_classes, anchors=anchors, training=True)
    elif args.net_type == 'yolov5m':
        net = my_yolov5m(testDataSet.num_classes, anchors=anchors, training=True)
    else:
        net = my_yolov5x(testDataSet.num_classes, anchors=anchors, training=True)
    net.to(device)
    net.load_state_dict(torch.load(args.pth, map_location=device))
    with torch.no_grad():
        precision, recall, AP, f1, ap_class, map, test_loss = evaluate(net, dataloaderTest, args.iou_thres, args.conf_thres, args.nms_thres, args.imgsize)
    print('MAP:', map)