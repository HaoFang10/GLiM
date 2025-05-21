python -u dev_test_data.py \
       --dataset chemdisgene \
       --transformer_type bert \
       --model_name_or_path BiomedBERT-large-uncased-abstract \
       --dev_file valid.json \
       --test_file test.anno_all.json \
       --load_path ../trained_model/model_REGT_save.pth \
       --test_batch_size 4 \
       --seed 66 \
       --num_class 15 \
       --tau 0.2 \
       --gpu 0\
       --ratio 0.5470931041421813 \
       --k 0.6

python infer.py
python merge.py
python eval.py
