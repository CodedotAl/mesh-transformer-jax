import argparse

from pathlib import Path
from subprocess import CalledProcessError, check_output, Popen

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_dir",
    type=str,
    help="Path to where your files are located. Files ending in .zst are "
    "treated as archives, all others as raw text.",
)
parser.add_argument(
    "--output_dir", type=str, default="./tfrecords", help="Where to put tfrecords"
)
parser.add_argument(
    "--split_size",
    type=int,
    default=100,
    help="Size of each split to be",
)

args = parser.parse_args()
input_dir = Path(args.input_dir)
output_dir = Path(args.output_dir)

# read in all the jsonl.zst files
files = list(sorted(input_dir.glob("*jsonl.zst")))

# group files into split_size chunks
split_files = [
    files[i : i + args.split_size] for i in range(0, len(files), args.split_size)
]


# create folders for each split and move the files into them
for i, split in enumerate(split_files):
    split_dir = input_dir / f"split_{i}"
    print(split_dir)
    split_dir.mkdir(exist_ok=True)
    for file in split:
        file.rename(split_dir / file.name)

    split_output_dir = output_dir / f"split_{i}"
    split_output_dir.mkdir(exist_ok=True, parents=True)
    # convert the files to tfrecords
    try:
        output = Popen(
            [
                "python",
                "/scratch/nacooper/mesh-transformer-jax/create_finetune_tfrecords.py",
                "--output_dir",
                str(split_output_dir.absolute()),
                "--input_dir",
                str(split_dir.absolute()),
                "--model_name",
                "gpt2",
                "--log_idt",
                "train",
            ],
            cwd="/scratch/nacooper/",
        )
        print(output)
    except CalledProcessError as e:
        print(e)
        print(e.output)
        raise e

# check_output(
#     [
#         "bash",
#         "clone.sh",
#         "repo_names.txt",
#     ],
#     cwd=str(out_path / "repos-commits" / "repos-commits"),
# )

# cd ..
# python mesh-transformer-jax/create_finetune_tfrecords.py --output_dir "records_data_dir/train" --model_name "gpt2" --log_idt "train" --input_dir "raw_data_dir/train"  #"code_clippy"
# echo "train split records generated..."
# python mesh-transformer-jax/create_finetune_tfrecords.py --output_dir "records_data_dir/validation" --model_name "gpt2" --log_idt "validation"  --input_dir "raw_data_dir/validation"  #"code_clippy"
# echo "eval split records generated..."
# python mesh-transformer-jax/create_finetune_tfrecords.py --output_dir "records_data_dir/test" --model_name "gpt2" --log_idt "test"  --input_dir "raw_data_dir/test" # "code_clippy"
# echo "test split records generated..."
