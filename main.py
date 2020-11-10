import glog as log

from engine.trainer import Trainer
from engine.tester import Tester
from utils.config import get_cfg_defaults

if __name__ == '__main__':
    cfg = get_cfg_defaults()
    # cfg.merge_from_file("./wandb/run-20201023_213704-3o2q3c4r/config.yaml")
    # cfg.freeze()
    print(cfg)

    if cfg.MODEL.IS_TRAIN:
        trainer = Trainer(cfg)
        trainer.run()
    else:
        tester = Tester(cfg)
        for i_id in list(range(101, 111)):
            for c_i_id in list(range(1010, 1020)):
                for mode in list(range(1, 9)):
                    tester.infer(mode=mode, img_id=i_id, c_img_id=c_i_id)
                    log.info("I: {}, C: {}, Mode:{}".format(i_id, c_i_id, mode))
