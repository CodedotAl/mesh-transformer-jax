cd ..
python mesh-transformer-jax/create_finetune_tfrecords.py --output_dir "records_data_dir/train" --model_name "gpt2" --log_idt "train" --input_dir "raw_data_dir/train"  #"code_clippy"
echo "train split records generated..."
python mesh-transformer-jax/create_finetune_tfrecords.py --output_dir "records_data_dir/validation" --model_name "gpt2" --log_idt "validation"  --input_dir "raw_data_dir/validation"  #"code_clippy"
echo "eval split records generated..."
python mesh-transformer-jax/create_finetune_tfrecords.py --output_dir "records_data_dir/test" --model_name "gpt2" --log_idt "test"  --input_dir "raw_data_dir/test" # "code_clippy"
echo "test split records generated..."