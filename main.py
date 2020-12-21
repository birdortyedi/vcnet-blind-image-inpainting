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

parser.add_argument("--tune", action="store_true", help="true for starting tune for ablation studies")

parser.add_argument("--test", "-t", action="store_true", help="true for testing phase")
parser.add_argument("--ablation", "-a", action="store_true", help="true for ablation studies")
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
    cfg.MODEL.IS_TRAIN = not args.test
    cfg.TRAIN.TUNE = args.tune
    # cfg.DATASET.NAME = args.dataset
    # cfg.DATASET.ROOT = args.dataset_dir
    # cfg.DATASET.CONT_ROOT = args.cont_dataset_dir
    # cfg.DATASET.IMAGENET = args.imagenet
    cfg.TEST.WEIGHTS = args.weights
    # cfg.TEST.ABLATION = args.ablation
    # cfg.TEST.MODE = args.test_mode
    # cfg.freeze()
    print(cfg)

    if cfg.MODEL.IS_TRAIN:
        trainer = Trainer(cfg)
        trainer.run()
    else:
        tester = Tester(cfg)
        if cfg.TEST.ABLATION:
            for i_id in list(range(250, 500)):
                for c_i_id in list(range(185, 375)):
                    for mode in list(range(1, 9)):
                        tester.do_ablation(mode=mode, img_id=i_id, c_img_id=c_i_id)
                        log.info("I: {}, C: {}, Mode:{}".format(i_id, c_i_id, mode))
        else:
            # qualitative
            img_path = "datasets/ffhq/images1024x1024/01000/01068.png"
            cont_path = "datasets/CelebAMask-HQ/CelebA-HQ-img/142.jpg"
            in_cont_path = "datasets/ImageNet/ILSVRC2012_img_val/ILSVRC2012_val_00000001.JPEG"
            mask_path = "../../Downloads/mask.jpg"
            graf_path = "../../Downloads/graf2.png"
            graf_mask_path = "../../Downloads/graf-mask-2.jpeg"
            # tester.infer(img_path, cont_path, mode=1)
            # tester.infer(img_path, in_cont_path, mask_path=mask_path, mode=1)
            # tester.infer(img_path, mask_path=mask_path, mode=2)
            # tester.infer(img_path, mode=3, mask_path=mask_path, color="RED")
            # tester.infer(img_path, mode=3, mask_path=mask_path, color="BLUE")
            # tester.infer(img_path, mode=3, mask_path=mask_path, color="GREEN")
            # tester.infer(img_path, mode=3, mask_path=mask_path, color="WHITE")
            # tester.infer(img_path, mode=3, mask_path=mask_path, color="BLACK")

            # tester.infer(img_path, mode=4) # ???
            # tester.infer(img_path, graf_path, mask_path=graf_mask_path, mode=5)
            # tester.infer(img_path, cont_path, mode=6)  # problematic
            # tester.infer(img_path, cont_path, mode=7, text="furkan", color="BLUE")  # problematic
            # tester.infer(img_path, cont_path, mode=8)  # problematic

            # quantitative
            tester.eval()
