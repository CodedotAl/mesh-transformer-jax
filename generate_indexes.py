import argparse

from pathlib import Path

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

args = parser.parse_args()
input_dir = Path(args.input_dir)
root_dir = input_dir.name
output_dir = Path(args.output_dir)

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

with open(output_dir / "code_clippy.train.index", "w") as f:
    f.write("\n".join(train_indexes))

with open(output_dir / "code_clippy.val.index", "w") as f:
    f.write("\n".join(val_indexes))
