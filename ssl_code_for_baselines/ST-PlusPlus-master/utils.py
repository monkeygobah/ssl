import numpy as np
from PIL import Image
import torch
from torchvision.models.segmentation import deeplabv3_resnet101

def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6

class meanIOU:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        return iu, np.nanmean(iu)


def color_map(dataset='pascal'):
    cmap = np.zeros((256, 3), dtype='uint8')

    if dataset == 'pascal' or dataset == 'coco':
        def bitget(byteval, idx):
            return (byteval & (1 << idx)) != 0

        for i in range(256):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7-j)
                g = g | (bitget(c, 1) << 7-j)
                b = b | (bitget(c, 2) << 7-j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

    elif dataset == 'cityscapes':
        cmap[0] = np.array([128, 64, 128])
        cmap[1] = np.array([244, 35, 232])
        cmap[2] = np.array([70, 70, 70])
        cmap[3] = np.array([102, 102, 156])
        cmap[4] = np.array([190, 153, 153])
        cmap[5] = np.array([153, 153, 153])
        cmap[6] = np.array([250, 170, 30])
        cmap[7] = np.array([220, 220, 0])
        cmap[8] = np.array([107, 142, 35])
        cmap[9] = np.array([152, 251, 152])
        cmap[10] = np.array([70, 130, 180])
        cmap[11] = np.array([220, 20, 60])
        cmap[12] = np.array([255,  0,  0])
        cmap[13] = np.array([0,  0, 142])
        cmap[14] = np.array([0,  0, 70])
        cmap[15] = np.array([0, 60, 100])
        cmap[16] = np.array([0, 80, 100])
        cmap[17] = np.array([0,  0, 230])
        cmap[18] = np.array([119, 11, 32])

    return cmap




def init_basic_elems(args, pretrained_path=None):
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Selected device: {DEVICE}')

    # Load DeepLabV3 with ResNet101 backbone and pretrained weights
    model = deeplabv3_resnet101(pretrained=True)
    model.classifier[4] = torch.nn.Conv2d(256, 6, kernel_size=(1, 1), stride=(1, 1))  # Adjust for 6 classes
    model.to(DEVICE)

    # Load pretrained weights if specified
    if pretrained_path:
        print(f"Loading pretrained model from {pretrained_path}")
        model.load_state_dict(torch.load(pretrained_path, map_location=DEVICE))

    # Adam Optimizer (matching Mean Teacher setup)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    return model, optimizer

# def init_basic_elems(args):

#     model_zoo = {'deeplabv3plus': DeepLabV3Plus, 'pspnet': PSPNet, 'deeplabv2': DeepLabV2}
#     model = model_zoo[args.model](args.backbone, 21 if args.dataset == 'pascal' else 19)

#     head_lr_multiple = 10.0
#     if args.model == 'deeplabv2':
#         assert args.backbone == 'resnet101'
#         model.load_state_dict(torch.load('pretrained/deeplabv2_resnet101_coco_pretrained.pth'))
#         head_lr_multiple = 1.0

#     optimizer = SGD([{'params': model.backbone.parameters(), 'lr': args.lr},
#                      {'params': [param for name, param in model.named_parameters()
#                                  if 'backbone' not in name],
#                       'lr': args.lr * head_lr_multiple}],
#                     lr=args.lr, momentum=0.9, weight_decay=1e-4)

#     model = DataParallel(model).cuda()

#     return model, optimizer