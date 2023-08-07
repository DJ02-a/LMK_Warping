import os
import socket

import cv2
import wandb


def setting_config(CONFIG):
    user_name = os.path.expanduser("~/").split("/")[-2]
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(("pwnbit.kr", 443))
    server_num = sock.getsockname()[0].split(".")[-1]

    wandb.config.update(
        {
            "BASIC | user_name": user_name,
            "BASIC | BATCH_PER_GPU": CONFIG["BASE"]["BATCH_PER_GPU"],
            "BASIC | SERVER_NUM": server_num,
            "OPTIMIZER | type": CONFIG["OPTIMIZER"]["TYPE"],
            "OPTIMIZER | lr_G": CONFIG["OPTIMIZER"]["LR_G"],
            "OPTIMIZER | lr_D": CONFIG["OPTIMIZER"]["LR_D"],
            "OPTIMIZER | BETA": CONFIG["OPTIMIZER"]["BETA"],
        }
    )

    # LOSS
    LOSS_DICT = {}
    for key, value in CONFIG["LOSS"].items():
        if value:
            LOSS_DICT["LOSS | " + key] = value
    wandb.config.update(LOSS_DICT)
