# mT5: Massively Multilingual Pre-trained Text-to-Text Transformer 

## Instructions to run the code:
You can access the models, their training and validation in the following Colab files:

Training vanilla mT5. Currently it is in Gujarati: [Training of mT5](https://colab.research.google.com/drive/1TDdo58cIKl4vrDjhgtX5CBZ5lbFPU9jd?usp=sharing)

Predicting summary by mT5.
[Predicting mT5](https://colab.research.google.com/drive/1BLPfMinTlLz9ttwKV4VHKNCVk069T0iZ?usp=sharing)

[This](https://drive.google.com/drive/folders/13HXeMVUhky1nJxsGO-W2eI-cEPHnsYe3?usp=sharing) contains fine-tuned Gujarati Summarisation model (trained up to 3 epochs).

If you want to run it locally, you may follow the following steps:

1. The dataset must be downloaded from here: \
[Hindi & English Dataset](https://drive.google.com/file/d/1PMquHwtPC_lbeorJgXd7fJsCT8008QhC/view?usp=sharing) \
[Gujarati Dataset](https://drive.google.com/file/d/1hiHwpTNMG-jcj3n-wDgzjJ2tOptcWfAL/view?usp=sharing) \
[XL-Sum](https://huggingface.co/datasets/csebuetnlp/xlsum)

2. Execute:
```
python3 pipeline.py --model_name_or_path "google/mt5-base" --data_dir "test/individual/hin" --output_dir "/scratch/devesh.marwah/Output" --lr_scheduler_type="linear" --learning_rate=5e-4 --warmup_steps 100 --weight_decay 0.01 \--per_device_train_batch_size=2 --gradient_accumulation_steps=16  --num_train_epochs=10 --save_steps 100 --predict_with_generate --evaluation_strategy "epoch" --logging_first_step --adafactor --label_smoothing_factor 0.1 --do_train --do_eval
```