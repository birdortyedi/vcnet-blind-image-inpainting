import math

from yacs.config import CfgNode as CN

_C = CN()

_C.SYSTEM = CN()
_C.SYSTEM.NUM_GPU = 1
_C.SYSTEM.NUM_WORKERS = 8

_C.WANDB = CN()
_C.WANDB.PROJECT_NAME = "vcnet-blind-image-inpainting"
_C.WANDB.ENTITY = "vvgl-ozu"
_C.WANDB.RUN = 11
_C.WANDB.LOG_DIR = ""
_C.WANDB.NUM_ROW = 0

_C.TRAIN = CN()
_C.TRAIN.NUM_TOTAL_STEP = 320000
_C.TRAIN.START_STEP = 0
_C.TRAIN.BATCH_SIZE = 4
_C.TRAIN.SHUFFLE = True
_C.TRAIN.NUM_STEPS_FOR_JOINT = 160000
_C.TRAIN.LOG_INTERVAL = 200
_C.TRAIN.SAVE_INTERVAL = 10000
_C.TRAIN.SAVE_DIR = "./weights"
_C.TRAIN.RESUME = False
_C.TRAIN.VISUALIZE_INTERVAL = 200

_C.MODEL = CN()
_C.MODEL.NAME = "VCNet"
_C.MODEL.IS_TRAIN = True

_C.MODEL.MPN = CN()
_C.MODEL.MPN.NAME = "MaskPredictionNetwork"
_C.MODEL.MPN.NUM_CHANNELS = 64
_C.MODEL.MPN.NECK_CHANNELS = 128
_C.MODEL.MPN.LR = 1e-4
_C.MODEL.MPN.BETAS = (0.5, 0.9)
_C.MODEL.MPN.SCHEDULER = []
_C.MODEL.MPN.DECAY_RATE = 0.
_C.MODEL.MPN.LOSS_COEFF = 2.

_C.MODEL.RIN = CN()
_C.MODEL.RIN.NAME = "RobustInpaintingNetwork"
_C.MODEL.RIN.NUM_CHANNELS = 32
_C.MODEL.RIN.NECK_CHANNELS = 128
_C.MODEL.RIN.LR = 1e-4
_C.MODEL.RIN.BETAS = (0.5, 0.9)
_C.MODEL.RIN.SCHEDULER = []
_C.MODEL.RIN.DECAY_RATE = 0.
_C.MODEL.RIN.LOSS_COEFF = 1.
_C.MODEL.RIN.EMBRACE = True

_C.MODEL.D = CN()
_C.MODEL.D.NAME = "1-ChOutputDiscriminator"
_C.MODEL.D.NUM_CHANNELS = 64
_C.MODEL.D.LR = 1e-3
_C.MODEL.D.BETAS = (0.5, 0.9)
_C.MODEL.D.SCHEDULER = [16, 24]
_C.MODEL.D.DECAY_RATE = 0.5
_C.MODEL.D.NUM_CRITICS = 5

_C.MODEL.JOINT = CN()
_C.MODEL.JOINT.NAME = "JointNetwork"
_C.MODEL.JOINT.LR = 1e-5
_C.MODEL.JOINT.BETAS = (0.5, 0.9)
_C.MODEL.JOINT.SCHEDULER = [24, ]
_C.MODEL.JOINT.DECAY_RATE = 0.5

_C.OPTIM = CN()
_C.OPTIM.GP = 10
_C.OPTIM.MASK = 1
_C.OPTIM.RECON = 1.4
_C.OPTIM.SEMANTIC = 1e-4
_C.OPTIM.TEXTURE = 1e-3
_C.OPTIM.ADVERSARIAL = 1e-3

_C.DATASET = CN()
_C.DATASET.NAME = "Places"  # "FFHQ"  # "Places"
_C.DATASET.ROOT = "./datasets/Places/imgs"  # "./datasets/ffhq/images1024x1024"  # "./datasets/Places/imgs"
_C.DATASET.CONT_ROOT = "./datasets/ImageNet/"  # "./datasets/CelebAMask-HQ"  # "./datasets/ImageNet/"
_C.DATASET.IMAGENET = "./datasets/ImageNet/"
_C.DATASET.SIZE = 256
_C.DATASET.MEAN = [0.5, 0.5, 0.5]
_C.DATASET.STD = [0.5, 0.5, 0.5]

_C.MASK = CN()
_C.MASK.MIN_NUM_VERTEX = 0
_C.MASK.MAX_NUM_VERTEX = 16
_C.MASK.MEAN_ANGLE = 2 * math.pi / 5
_C.MASK.ANGLE_RANGE = 2 * math.pi / 15
_C.MASK.MIN_WIDTH = 8
_C.MASK.MAX_WIDTH = 20
_C.MASK.NUM_ITER_SMOOTHING = 3
_C.MASK.MIN_REMOVAL_RATIO = 0.2
_C.MASK.MAX_REMOVAL_RATIO = 0.5
_C.MASK.GAUS_K_SIZE = 15
_C.MASK.SIGMA = 4

_C.TEST = CN()
_C.TEST.OUTPUT_DIR = "./outputs"
_C.TEST.ITER = 4
_C.TEST.MODE = 7
_C.TEST.IMG_ID = 52
_C.TEST.C_IMG_ID = 38
_C.TEST.BRUSH_COLOR = "RED"
_C.TEST.GRAFFITI_PATH = "./datasets/graffiti-dataset/dataset/graffiti_sample/000001ff0013ffff.p"
_C.TEST.TEXT = "hello world"
_C.TEST.FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
_C.TEST.FONT_SIZE = 24


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()


# provide a way to import the defaults as a global singleton:
cfg = _C  # users can `from config import cfg`
