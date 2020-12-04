import glog as log
import argparse
from engine.trainer import Trainer
from engine.tester import Tester
from utils.config import get_cfg_defaults

parser = argparse.ArgumentParser()

parser.add_argument("--base_cfg", default="./wandb/run-20201023_213704-3o2q3c4r/config.yaml", metavar="FILE", help="path to config file")
parser.add_argument("--weights", "-w", default=None, help="weights for VCNet")
parser.add_argument("--dataset", "-d", default="FFHQ", help="dataset names: FFHQ, Places")
parser.add_argument("--dataset_dir", default="./datasets/ffhq/images1024x1024", help="dataset directory: './datasets/ffhq/images1024x1024', "
                                                                                     " './datasets/Places/imgs'")
parser.add_argument("--cont_dataset_dir", default="./datasets/CelebAMask-HQ", help="contaminant dataset directory: './datasets/CelebAMask-HQ', "
                                                                                   " './datasets/ImageNet/'")
parser.add_argument("--imagenet", default="./datasets/ImageNet/", help="imagenet directory: './datasets/ImageNet/'")
parser.add_argument("--test", action="store_true", help="true for testing phase")
parser.add_argument("--test_mode", default=1, help="test mode: 1: contaminant image,"
                                                   "2: random brush strokes with noise,"
                                                   "3: random brush strokes with colors,"
                                                   "4: real occlusions,"
                                                   "5: graffiti,"
                                                   "6: facades,"
                                                   "7: words,"
                                                   "8: face swap")

args = parser.parse_args()

if __name__ == '__main__':
    cfg = get_cfg_defaults()
    # cfg.merge_from_file(args.base_cfg)
    # cfg.MODEL.IS_TRAIN = not args.test
    # cfg.DATASET.NAME = args.dataset
    # cfg.DATASET.ROOT = args.dataset_dir
    # cfg.DATASET.CONT_ROOT = args.cont_dataset_dir
    # cfg.DATASET.IMAGENET = args.imagenet
    # cfg.TEST.WEIGHTS = args.weights
    # cfg.TEST.MODE = args.test_mode
    # cfg.freeze()
    print(cfg)

    if cfg.MODEL.IS_TRAIN:
        trainer = Trainer(cfg)
        trainer.run()
    else:
        tester = Tester(cfg)
        for i_id in list(range(1001, 1111)):
            for c_i_id in list(range(2010, 2020)):
                for mode in list(range(1, 9)):
                    tester.infer(mode=mode, img_id=i_id, c_img_id=c_i_id)
                    log.info("I: {}, C: {}, Mode:{}".format(i_id, c_i_id, mode))
