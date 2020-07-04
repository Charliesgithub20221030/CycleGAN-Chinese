python3 main.py \
    -train \
    -mode pretrain \
    -model_dir 'data/output_model/ptt_model_v2' \
    -dict 'word2vec_model/dictionary_ptt_word2vec.json' \
    -data_source 'data/feature_ptt_spliit.txt' \
    -num_steps 60000
