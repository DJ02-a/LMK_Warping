import os
import sys
import traceback

sys.path.append("./")
sys.path.append("./lib/discriminators/")
import warnings
from distutils import dir_util

import torch
import wandb

from core.model import MyModel
from lib import utils

warnings.filterwarnings("ignore")
from datetime import datetime, timedelta

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import InitProcessGroupKwargs
from PIL import Image

sys.path.append("./.utils")
from wandb_utils import setting_config


def train():
    # load config
    CONFIG = utils.load_jsonnet("./configs.jsonnet")
    sys.path.append(CONFIG["BASE"]["PACKAGES_PATH"])

    # update configs
    CONFIG["BASE"]["GPU_NUM"] = torch.cuda.device_count()
    CONFIG["BASE"]["GLOBAL_STEP"] = 0

    accelerator = Accelerator(
        log_with="wandb",
        kwargs_handlers=[
            DistributedDataParallelKwargs(
                find_unused_parameters=False,
            ),
            InitProcessGroupKwargs(timeout=timedelta(hours=3)),
        ],
    )

    if accelerator.is_main_process:
        # save config
        utils.make_dirs(CONFIG)
        utils.print_dict(CONFIG)
        utils.save_json(
            f"{CONFIG['BASE']['SAVE_ROOT_RUN']}/config_{CONFIG['BASE']['RUN_ID']}",
            CONFIG,
        )
        dir_util.copy_tree("./core", CONFIG["BASE"]["SAVE_ROOT_CODE"])

    model = MyModel(CONFIG, accelerator)

    model.accelerator.init_trackers(
        CONFIG["BASE"]["MODEL_ID"],
        init_kwargs={
            "wandb": {"name": CONFIG["BASE"]["RUN_ID"], "tags": CONFIG["TAGS"]}
        },
    )  # @#
    setting_config(CONFIG)
    # Training loop
    try:
        while CONFIG["BASE"]["GLOBAL_STEP"] < CONFIG["BASE"]["MAX_STEP"]:
            # go one step
            model.go_step()
            model.accelerator.wait_for_everyone()

            if model.accelerator.is_main_process:
                # Save and print loss
                if CONFIG["BASE"]["GLOBAL_STEP"] % CONFIG["CYCLE"]["LOSS"] == 0:
                    model.loss_collector.print_loss()
                    model.accelerator.log(model.loss_collector.loss_dict)

                # Save image
                if CONFIG["BASE"]["GLOBAL_STEP"] % CONFIG["CYCLE"]["TRAIN_IMAGE"] == 0:
                    train_images_grid = utils.make_grid_image(model.train_images)
                    train_images_grid.save(
                        f"{CONFIG['BASE']['SAVE_ROOT_IMGS']}/{str(CONFIG['BASE']['GLOBAL_STEP']).zfill(8)}_train.png"
                    )
                    train_images_grid.save(
                        f"{CONFIG['BASE']['SAVE_ROOT_IMGS']}/_latest_train_result.png"
                    )

                if CONFIG["BASE"]["GLOBAL_STEP"] % CONFIG["CYCLE"]["VALID_IMAGE"] == 0:
                    model.do_validation()

                    valid_images_grid = utils.make_grid_image(model.valid_images)
                    valid_images_grid.save(
                        f"{CONFIG['BASE']['SAVE_ROOT_IMGS']}/{str(CONFIG['BASE']['GLOBAL_STEP']).zfill(8)}_valid.png"
                    )
                    valid_images_grid.save(
                        f"{CONFIG['BASE']['SAVE_ROOT_IMGS']}/_latest_valid_result.png"
                    )

                # Save checkpoint parameters
                if CONFIG["BASE"]["GLOBAL_STEP"] % CONFIG["CYCLE"]["CKPT"] == 0:
                    model.save_checkpoint()

                if CONFIG["BASE"]["WANDB_STEP"] % CONFIG["CYCLE"]["WANDB"] == 0:
                    wandb.log({"TRAIN": wandb.Image(train_images_grid)})
                    wandb.log({"VALID": wandb.Image(valid_images_grid)})

            CONFIG["BASE"]["GLOBAL_STEP"] += 1
        model.accelerator.end_training()

    except Exception as ex:
        err_msg = traceback.format_exc()
        now = datetime.now()
        wandb.alert(
            "Exception occured",
            f"[ TIME ]\n{now.strftime('%Y-%m-%d %H:%M:%S')} \n[ ITERS ]\n{CONFIG['BASE']['GLOBAL_STEP']} \n\n[ ERROR_MSG ]\n{err_msg}",
        )


if __name__ == "__main__":
    train()
