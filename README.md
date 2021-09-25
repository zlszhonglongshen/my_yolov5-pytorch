# my_yolov5-pytorch
这一套程序包含yolov5的训练和测试以及推理目标检测可视化的功能，支持yolov5s,yolov5l,yolov5x,yolov5m四种结构的yolo
并且yolo的网络结构都是在.py文件里定义的，这是与ultralytics发布的yolov5的一个最大不同之处。
此外，在训练或者测试的时候，配置参数都是argparse.ArgumentParser()传入的。
整套程序只有11个.py文件，不包含子文件夹。为了验证程序的有效性，我用yolov5s在voc2012数据集上训练完300个epoch。

## 训练步骤：

本套程序在训练时的标签文件支持xml和json格式的，训练的主程序是 main_train.py

在检查确认训练数据集和标签文件无误之后，以pascal_voc2012数据集为例，在终端输入

```
python main_train.py --imgroot=/home/data/datasets/VOCdevkit/VOC2012/JPEGImages --labroot=/home/data/datasets/VOCdevkit/VOC2012/Annotations --train_txt=/home/data/datasets/VOCdevkit/VOC2012/ImageSets/Main/trainval.txt --test_txt=/home/data/datasets/VOCdevkit/VOC2012/ImageSets/Main/val.txt --labels_txt=/home/data/datasets/VOCdevkit/pascal_voc.names --save_model=pascal_voc --epochs=300 --batchsize=16 --plot_loss --write_excel
```

就可以启动训练了。如果在新数据集上训练，那么需要输入新的配置参数

## 测试步骤：

主程序是main_test.py，以pascal_voc2012数据集为例，在终端输入
```
python main_train.py --imgroot=/home/data/datasets/VOCdevkit/VOC2012/JPEGImages --labroot=/home/data/datasets/VOCdevkit/VOC2012/Annotations --test_txt=/home/data/datasets/VOCdevkit/VOC2012/ImageSets/Main/val.txt --labels_txt=/home/data/datasets/VOCdevkit/pascal_voc.names
```
程序运行完后会打印出map,precision,recall,loss等信息

## 推理：

主程序是detect_img.py，以pascal_voc2012数据集为例，在终端输入
```
python detect_img.py --img_path=/home/data/datasets/VOCdevkit/VOC2012/JPEGImages/2008_003261.jpg
```

程序运行完后会在窗口画出目标矩形框和类别
