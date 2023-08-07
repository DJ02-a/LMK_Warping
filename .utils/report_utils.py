import glob
import json
import math
import os
import pdb
import shutil
import sys
from datetime import datetime

import _jsonnet
import yaml
from github import Github
from pytz import timezone


def get_file_size(size_bytes):
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = str(round(size_bytes / p, 2))
    return s + ' ' + size_name[i]

def prepare_issue(global_step, run_id):
    latest_image_path = os.path.join(
        f"./train_results/{run_id}/imgs/_latest_train_result.png"
    )
    os.makedirs(f"./.utils/reports/{run_id}", exist_ok=True)
    shutil.copy(
        latest_image_path, f"./.utils/reports/{run_id}/_latest_train_result.png"
    )

    with open("./wandb/latest-run/files/config.yaml") as f:
        CONFIG = yaml.load(f, Loader=yaml.FullLoader)
        CONFIG['BASIC']['value']['GLOBAL_STEP'] = global_step
        
    with open("./wandb/latest-run/files/config.yaml", 'w') as f:
        yaml.dump(CONFIG, f)

def set_basic_infos(metadata, CONFIG):
    contents = ''
    contents += '# INFOS\n'
    contents += '## Run Infos\n\n'
    contents += '```python\n'
    contents += '{0:^15} | {1}\n'.format('HOST', str(metadata["host"]))
    contents += '{0:^15} | {1}\n'.format('USERNAME', str(metadata["username"]))
    
    memory = get_file_size(metadata["gpu_devices"][0]["memory_total"])
    contents += '{0:^15} | {1}\n'.format('GPU', str(metadata["gpu_devices"][0]["name"]))
    contents += '{0:^15} | {1}\n'.format('GPU_NUM', str(CONFIG["BASE"]["value"]["GPU_NUM"]))
    contents += '{0:^15} | {1}\n'.format('GPU_MEM', str(memory))
    contents += '{0:^15} | {1}\n'.format('FILE', str(metadata["program"]))
    contents += '```\n'
    
    contents += '## Train Infos\n'
    contents += '```python\n'
    contents += '{0:^15} | {1}\n'.format('MODEL_NAME', str(CONFIG["BASE"]["value"]["MODEL_ID"]))
    contents += '{0:^15} | {1}\n'.format('RUN_ID', str(CONFIG["BASE"]["value"]["RUN_ID"]))
    contents += '{0:^15} | {1}\n'.format('IMG_SIZE', str(CONFIG["BASE"]["value"]["IMG_SIZE"]))
    contents += '{0:^15} | {1}\n'.format('BATCH_SIZE', str(CONFIG["BASE"]["value"]["BATCH_PER_GPU"]))
    contents += '{0:^15} | {1}\n'.format('STEPS', str(CONFIG["BASE"]["value"]["GLOBAL_STEP"]))
    contents += '```\n'
    
    # contents += '{0:^15} | {1}\n'.format('WANDB_URL', wandb_url)
    return contents

def get_table(loss):
    index, line, weight = "|", "|", "|"
    for key, value in loss.items():
        if value:
            index += key + "|"
            line += ":---:|"
            weight += str(value) + "|"

    table_str = index + "\n" + line + "\n" + weight + "\n"
    return table_str


def get_github_repo(access_token, repository_name):
    """
    github repo object를 얻는 함수`
    :param access_token: Github access token
    :param repository_name: repo 이름
    :return: repo object
    """
    g = Github(access_token)
    return g.get_user().get_repo(repository_name)


def upload_github_issue(repo, title, body, labels):
    """
    해당 repo에 title 이름으로 issue를 생성하고, 내용을 body로 채우는 함수
    :param repo: repo 이름
    :param title: issue title
    :param body: issue body
    :return: None
    """
    repo.create_issue(
        title=title,
        body=body,
        labels=labels,
    )


# def
# 여기에 report image 칸을 만들어 놓자

if __name__ == "__main__":
    # basic info
    access_token = os.environ["MY_GITHUB_TOKEN"]
    wandb_id = os.environ["WANDB_ID"]
    github_id = os.environ["GITHUB_ID"]

    seoul_timezone = timezone("Asia/Seoul")
    today = datetime.now(seoul_timezone)
    today_date = today.strftime("%Y년 %m월 %d일")

    # get run info
    with open("./wandb/latest-run/files/wandb-metadata.json") as json_file:
        metadata = json.load(json_file)
    with open("./wandb/latest-run/files/config.yaml") as f:
        CONFIG = yaml.load(f, Loader=yaml.FullLoader)
        
    repository_name = CONFIG["BASE"]["value"]["MODEL_ID"]
    run_id = CONFIG["BASE"]["value"]["RUN_ID"]
    wandb_run_id = [file_name for file_name in os.listdir('./wandb/latest-run/') if file_name.endswith(r'.wandb')][0].split('-')[-1].split('.')[0]
    wandb_url = f"https://wandb.ai/{wandb_id}/{repository_name}/runs/{wandb_run_id}?workspace=user-{wandb_id}"

    
    issue_title = f"[ TRAIN ] {sys.argv[1]}"

    upload_contents = set_basic_infos(metadata, CONFIG)
    ## 
    upload_contents += '# RUN\n'
    upload_contents += 'wandb link : ' + wandb_url + '\n'
    upload_contents += '## Purpose\n'
    upload_contents += '- sample\n\n'
    upload_contents += '## Method\n'
    upload_contents += '- sample\n\n'
    
    ## table
    upload_contents += "## Loss\n"
    upload_contents += get_table(CONFIG["LOSS"]["value"])
    upload_contents += '\n\n'
    upload_contents += "## Optimization\n"
    upload_contents += get_table(CONFIG["OPTIMIZER"]["value"])
    
    # 결과, 관찰, 
    upload_contents += "# RESULT\n"
    upload_contents += "## Sample\n"
    upload_contents += f"<img src='https://raw.githubusercontent.com/{github_id}/{repository_name}/main/.utils/reports/{run_id}/_latest_train_result.png' width='10%' heigh='10%'>\n"
    upload_contents += '\n\n'
    upload_contents += "## Observation\n"
    upload_contents += '- sample\n\n'

    repo = get_github_repo(access_token, repository_name)
    
    labels = CONFIG["TAGS"]["value"]
    upload_github_issue(repo, issue_title, upload_contents, labels)

