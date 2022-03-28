"""
This script loads the data from Google Cloud Storage to a Huggingface dataset repository.
"""

import argparse

from pathlib import Path
import os
import time
import shutil
# from smart_open import open
# from google.cloud import storage
# from google.cloud.exceptions import NotFound

parser = argparse.ArgumentParser()
parser.add_argument(
    "--gs_project_id",
    type=str,
    help="Google Cloud Storage project ID",
    default = 'code-clippy-bucket'
)

parser.add_argument(
    "--output_dir", type=str, help="Where the files will be temprorarily stored locally.",  default = "../code_clippy_github/"
)


parser.add_argument(
    "--input_dir", type=str, help="Where the dataset is stored on the GCS.",  default = "code-clippy-dataset"
)


args = parser.parse_args()
root_dir = args.input_dir
output_dir = Path(args.output_dir)
# load_bucket = args.load_bucket

os.chdir(output_dir)

list_files = os.popen(f'gsutil ls -r gs://{args.gs_project_id}/{root_dir}').read().split('\n')

uploaded_files = os.popen(f'ls -r {output_dir}').read().split('\n')

#print(uploaded_files)
json_files_list = [f for f in list_files if '.json.gz' in str(f) and str(f.split('/')[-1]) not in uploaded_files]

print(json_files_list)

commited_files = []

for commit_num , file_path in enumerate(json_files_list,1):

    os.system( f'gsutil cp {file_path} {file_path.split("/")[-1]}')
    time.sleep(0.5)
    os.system( f'git add {file_path.split("/")[-1]}')

    commited_files.append(file_path.split("/")[-1])

    if commit_num % 20 == 0:
        time.sleep(1)
        os.system(f'git commit -m \" adding dataset from GCS {commit_num}\"')
        time.sleep(1)
        os.system(f'git push https://USERNAME:PASSWORD@huggingface.co/datasets/repo.git')

        time.sleep(1)

        while len(commited_files) > 0:
            os.remove(f'{commited_files.pop(0)}')

        print('Done Deleting')

    if commit_num % 200 == 0:
        os.chdir('..')
        shutil.rmtree(f'{str(output_dir).split("/")[-1]}')
       # os.remove(f'{str(output_dir).split("/")[-1]}')
        os.system(f'GIT_LFS_SKIP_SMUDGE=1 git clone https://USERNAME:PASSWORD@huggingface.co/datasets/CodedotAI/code_clippy_github.git')
        os.chdir(f'{str(output_dir).split("/")[-1]}')
        print(f'Completion: {commit_num/len(json_files_list) * 100} %')
