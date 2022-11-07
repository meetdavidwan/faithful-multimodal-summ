# Evaluating and Improving Factuality in Multimodal Abstractive Summarization (EMNLP 2022)

This repository contains PyTorch code for running the metric and training, as well as datasets and meta-evaluatation.

- Authors: [David Wan](https://meetdavidwan.github.io/) and [Mohit Bansal](https://www.cs.unc.edu/~mbansal/) (UNC Chapel Hill)

- Paper coming soon!

## Intallation and Dependencies
- Python 3.8
- PyTorch 1.10.2
- datasets 2.0.0
- transformers 4.17.0
- deepspeed 0.6.4
- nltk 3.7
- rouge_score 0.0.4
- clip 1.0
- bert_score 0.3.11

## Code

### 1. CLIPBERTScore

Please see `code/clipbertscore.py` for an example of running the metric. It essentially uses the original code of the corresponding metric.

If you would like to run the submodules separately:

To use CLIPScore, please set the weight `w=1`, or run the CLIP model directly.

For BERTScore, please follow [BERTScore](https://github.com/Tiiiger/bert_score) to install and run the corresponding metric. We use `BERTScorer( model_type="roberta-large-mnli", num_layers=10, device=device)`.

An alterantive implementation of CLIPBERTScore as a reward for rl can also be seen in `code/src/self_critical.py`.

### 2. Downstream Application
The code is found under `code/src` is adapted from Transformers' summarization example.

The directory contains the code for CLIP-BART and self-critical training.

The code assumes extracted CLIP features for the images. For an example of how to extract, please see `src/extract_features.py`

An example to run CLIP-BART on MMSS is:
```
python  src/run_summarization.py --fp16 \
--data_dir data/mmss --do_train --image_dir data/hidden_states/mmss/rn50x64 \
--summary_column summary --text_column document \
--model_name facebook/bart-base \
--tokenizer_name facebook/bart-base \
--do_eval --evaluation_strategy epoch  --predict_with_generate \
--per_device_train_batch_size 128 --per_device_eval_batch_size 16 \
--gradient_accumulation_steps 2 \
--learning_rate 3e-05 --weight_decay 0.01 --label_smoothing 0.1 \
--max_source_length 128 --max_target_length 32 \
--logging_step 100 --max_steps 5000 \
--warmup_steps 0 --save_steps 1000 --remove_unused_columns false \
--output_dir mmss_bart_base_rn50x64 --visual_feat_dim 1024
```

An example to run with rl:
```
python src/run_summarization.py --fp16 \
--data_dir data/mmss --do_train --image_dir data/hidden_states/mmss/rn50x64 \
--summary_column summary --text_column document \
--model_name mmss_bart_base_rn50x64 \
--do_eval --evaluation_strategy epoch --predict_with_generate  \
--per_device_train_batch_size 256 --per_device_eval_batch_size 16 \
--gradient_accumulation_steps 1 \
--learning_rate 3e-05 --weight_decay 0.01 \
--max_source_length 128 --max_target_length 32 \
--logging_step 100 --max_steps 5000 \
--warmup_steps 0 --save_steps 1000 --remove_unused_columns false \
--output_dir mmss_bart_base_rn50x64_rl_0.998_rouge2_cbs2 --visual_feat_dim 1024 \
--reward_model reward_model --train_self_critical --rl_weight 0.998 --rouge_key rouge2  --cbs_weight 2.0 --rouge_weight 1.0
```

## Data
Please see the `data` directory for all the relevant data files, including:

- MuFaME Meta-Evaluation
- WikiHowFact
- Multimodal WikiHow Summarization Datasets

Please refer to the corresponding directory for more details.
Preprocessed data coming soon!

# Reference
```BibTex
@inproceedings{wan2022evaluating,
      title={Evaluating and Improving Factuality in Multimodal Abstractive Summarization}, 
      author={Wan, David and Bansal, Mohit},
      booktitle={EMNLP 2022},
      year={2022}
}
```