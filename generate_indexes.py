import argparse

from pathlib import Path
import os
parser = argparse.ArgumentParser()
parser.add_argument(
    "--gs_project_id",
    type=str,
    help="Google Cloud Storage project ID",
)
parser.add_argument(
    "--input_dir", type=str, help="Where the tfrecords are stored locally. It must match the structure in the bucket."
)
parser.add_argument(
    "--output_dir", type=str, help="Where the indexes will be stored locally."
)

parser.add_argument ("--load_bucket", type=bool, help="Whether to load the folder structure from the Bucket directly", default = False, nargs = '?')

args = parser.parse_args()
input_dir = Path(args.input_dir)
root_dir = input_dir.name
output_dir = Path(args.output_dir)
load_bucket = args.load_bucket
# get the list of tfrecords
# train_tfrecords = [
#     str(f)
#     for f in input_dir.glob("*.tfrecords")
#     if "train" in str(f)
# ]
# val_tfrecords = [
#     str(f)
#     for f in input_dir.glob("*.tfrecords")
#     if "valid" in str(f)
# ]
# construct index file paths in the format of a gs bucket
if load_bucket  == False:
    train_indexes = [
                        f"gs://{args.gs_project_id}/{root_dir}/{str(f).split(f'/{root_dir}/')[-1]}"
                        for f in input_dir.glob("**/*.tfrecords")
                        if "train" in str(f)
                    ]
    print(train_indexes[:5])
    val_indexes = [
                        f"gs://{args.gs_project_id}/{root_dir}/{str(f).split(f'/{root_dir}/')[-1]}"
                        for f in input_dir.glob("**/*.tfrecords")
                        if "valid" in str(f)
                        ]

else:
    list_files = os.popen(f'gsutil ls -r gs://{args.gs_project_id}/{root_dir}').read().split('\n')
    train_indexes = [f for f in list_files if 'train' in str(f) and '.tfrecords' in str(f)]
    val_indexes = [f for f in list_files if 'valid' in str(f) and '.tfrecords' in str(f)]
with open(output_dir / "code_clippy.train.index", "w") as f:
    f.write("\n".join(train_indexes))

with open(output_dir / "code_clippy.val.index", "w") as f:
    f.write("\n".join(val_indexes))
