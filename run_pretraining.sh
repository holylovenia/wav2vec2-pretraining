################# CANTONESE PRE-TRAINING

# Wav2Vec 2.0 Cantonese
CUDA_VISIBLE_DEVICES=0 python pretrain.py --model_name_or_path=facebook/wav2vec2-large-xlsr-53 \
   --train_manifest_path=data/common_voice_zh-HK/preprocessed_validated_train.csv \
   --valid_manifest_path=data/common_voice_zh-HK/preprocessed_validated_dev.csv \
   --test_manifest_path=data/common_voice_zh-HK/preprocessed_validated_test.csv \
   --preprocessing_num_workers=16 --audio_column_name=path --text_column_name=sentence \
   --per_device_train_batch_size=8 --per_device_eval_batch_size=8 \
   --dataloader_num_workers=16 --dataloader_pin_memory --group_by_length \
   --seed=14045 --num_train_epochs=1000 --learning_rate=5e-5 \
   --fp16 --fp16_backend=amp \
   --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
   --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=10 \
   --save_strategy=epoch --save_steps=1 --save_total_limit=3 --load_best_model_at_end \
   --gradient_checkpointing=True \
   --metric_for_best_model="cer" \
   --greater_is_better=False \
   --cache_dir="pretrain/cache" \
   --output_dir="./pretrain/save" \
   --early_stopping_patience=10 \

################# CANTONESE PRE-TRAINING

# # Wav2Vec 2.0 Cantonese
# CUDA_VISIBLE_DEVICES=0 python pretrain.py --model_name_or_path=./pretrain/save/facebook/wav2vec2-large-xlsr-53_first-pretraining/checkpoint-10912 \
#    --resume_from_checkpoint=./pretrain/save/facebook/wav2vec2-large-xlsr-53/checkpoint-10912 \
#    --train_manifest_path=data/common_voice_zh-HK/preprocessed_train.csv \
#    --valid_manifest_path=data/common_voice_zh-HK/preprocessed_dev.csv \
#    --test_manifest_path=data/common_voice_zh-HK/preprocessed_test.csv \
#    --preprocessing_num_workers=16 --audio_column_name=path --text_column_name=sentence \
#    --per_device_train_batch_size=16 --per_device_eval_batch_size=16 \
#    --dataloader_num_workers=16 --dataloader_pin_memory --group_by_length \
#    --seed=14045 --num_train_epochs=100 --learning_rate=5e-5 \
#    --fp16 --fp16_backend=amp \
#    --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
#    --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=10 \
#    --save_strategy=epoch --save_steps=1 --save_total_limit=3 --load_best_model_at_end \
#    --gradient_checkpointing=True \
#    --cache_path="pretrain/cache/facebook/wav2vec2-large-xlsr-53" \
#    --output_path="./pretrain/save/facebook/wav2vec2-large-xlsr-53" \
#    --metric_for_best_model=cer --greater_is_better=False \
