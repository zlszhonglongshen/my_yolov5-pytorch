# my_yolov5-pytorch
这一套程序包含yolov5的训练和测试以及推理目标检测可视化的功能，支持yolov5s,yolov5l,yolov5x,yolov5m四种结构的yolo
并且yolo的网络结构都是在.py文件里定义的，这是与ultralytics发布的yolov5的一个最大不同之处。
此外，在训练或者测试的时候，配置参数都是argparse.ArgumentParser()传入的。
整套程序只有11个.py文件，不包含子文件夹。为了验证程序的有效性，我选择yolov5s在voc2012数据集上训练的，在测试集上的map等于0.476
模型pth文件上传在百度云盘，地址是
链接: https://pan.baidu.com/s/13_La0AcLwh7nYiE5yYYIrw  密码: k6vf
