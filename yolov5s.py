from common import *

class My_YOLO_backbone_head(nn.Module):
    def __init__(self, num_classes, num_anchors):
        super().__init__()
        self.no = (num_classes + 5) * num_anchors
        self.seq0_Focus = Focus(3, 32, 3)
        self.seq1_Conv = Conv(32, 64, 3, 2)
        self.seq2_Bottleneck = Bottleneck(64, 64)
        self.seq3_Conv = Conv(64, 128, 3, 2)
        self.seq4_BottleneckCSP = BottleneckCSP(128, 128, 3)
        self.seq5_Conv = Conv(128, 256, 3, 2)
        self.seq6_BottleneckCSP = BottleneckCSP(256, 256, 3)
        self.seq7_Conv = Conv(256, 512, 3, 2)
        self.seq8_SPP = SPP(512, 512, [5, 9, 13])
        self.seq9_BottleneckCSP = BottleneckCSP(512, 512, 2)
        self.seq10_BottleneckCSP = BottleneckCSP(512, 512, 1, False)
        self.seq11_Conv2d = nn.Conv2d(512, self.no, 1, 1, 0)
        self.seq14_Conv = Conv(768, 256, 1, 1)
        self.seq15_BottleneckCSP = BottleneckCSP(256, 256, 1, False)
        self.seq16_Conv2d = nn.Conv2d(256, self.no, 1, 1, 0)
        self.seq19_Conv = Conv(384, 128, 1, 1)
        self.seq20_BottleneckCSP = BottleneckCSP(128, 128, 1, False)
        self.seq21_Conv2d = nn.Conv2d(128, self.no, 1, 1, 0)
    def forward(self, x):
        x = self.seq0_Focus(x)
        x = self.seq1_Conv(x)
        x = self.seq2_Bottleneck(x)
        x = self.seq3_Conv(x)
        xRt0 = self.seq4_BottleneckCSP(x)
        x = self.seq5_Conv(xRt0)
        xRt1 = self.seq6_BottleneckCSP(x)
        x = self.seq7_Conv(xRt1)
        x = self.seq8_SPP(x)
        x = self.seq9_BottleneckCSP(x)
        route = self.seq10_BottleneckCSP(x)
        out0 = self.seq11_Conv2d(route)
        route2 = F.interpolate(route, size=(int(route.shape[2] * 2), int(route.shape[3] * 2)), mode='nearest')
        x = torch.cat([route2, xRt1], dim=1)
        x = self.seq14_Conv(x)
        route = self.seq15_BottleneckCSP(x)
        out1 = self.seq16_Conv2d(route)
        route2 = F.interpolate(route, size=(int(route.shape[2] * 2), int(route.shape[3] * 2)), mode='nearest')
        x = torch.cat([route2, xRt0], dim=1)
        x = self.seq19_Conv(x)
        x = self.seq20_BottleneckCSP(x)
        out2 = self.seq21_Conv2d(x)
        return out2, out1, out0

class My_YOLO(nn.Module):
    def __init__(self, num_classes, anchors=(), training=False):
        super().__init__()
        self.num_anchors = len(anchors[0]) // 2 
        self.backbone_head = My_YOLO_backbone_head(num_classes, self.num_anchors)
        self.yolo_layers = Yolo_Layers(num_classes, anchors=anchors, training=training)
    def forward(self, x):
        out2, out1, out0 = self.backbone_head(x)
        output = self.yolo_layers([out2, out1, out0])
        return output
