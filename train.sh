set -eux

pretrained_model=./pretrained_lm/roberta_base/
data_dir=./data/DocRED/

lr=5e-5
epoch=40
batch_size=4

python ./run_docred.py\
  --model_type roberta\
  --entity_structure biaffine\
  --model_name_or_path ./pretrained_lm/roberta_base/\
  --do_train\
  --do_eval\
  --data_dir data/DocRED/\
  --max_seq_length 512\
  --max_ent_cnt 42\
  --per_gpu_train_batch_size 4\
  --learning_rate 5e-5\
  --num_train_epochs 40\
  --warmup_ratio 0.1\
  --output_dir checkpoints\
  --seed 42\
  --logging_steps 10
