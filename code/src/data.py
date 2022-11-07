import os
import logging
import torch

import numpy as np

from datasets import load_from_disk, concatenate_datasets, DatasetDict
from data_collator import DataCollatorForSeq2SeqWithImages

logger = logging.getLogger(__name__)


class Data:
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer

        self.train_dataset = None
        self.eval_dataset = None
        self.predict_dataset = None

        self.image_size = 50

    def load_dataset(self, do_train=False, do_eval=False, do_predict=False):
        raw_datasets = load_from_disk(self.args.data_dir)

        # Preprocessing the datasets.
        # We need to tokenize inputs and targets.
        if do_train:
            column_names = raw_datasets["train"].column_names
        elif do_eval:
            column_names = raw_datasets["validation"].column_names
        elif do_predict:
            column_names = raw_datasets["test"].column_names
        else:
            logger.info(
                "There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`."
            )
            return
        
        # remove image from column names so it does not get deleted
        column_names.remove("images")
        if "image_label" in column_names:
            column_names.remove("image_label")

        self.padding = "max_length" if self.args.pad_to_max_length else False

        if do_train:
            if "train" not in raw_datasets:
                raise ValueError("--do_train requires a train dataset")
            train_dataset = raw_datasets["train"]
            if self.args.max_train_samples is not None:
                # train_dataset = train_dataset.shuffle(seed=42)
                train_dataset = train_dataset.select(range(self.args.max_train_samples))

            # Temporarily set max_target_length for training.
            self.max_target_length = self.args.max_target_length

            train_dataset = train_dataset.map(
                self.preprocess_function,
                batched=True,
                num_proc=self.args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not self.args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
            train_dataset.set_transform(self.transform_func)

            self.train_dataset = train_dataset

        if do_eval:
            self.max_target_length = self.args.val_max_target_length
            if "validation" not in raw_datasets:
                raise ValueError("--do_eval requires a validation dataset")
            eval_dataset = raw_datasets["validation"]

            if self.args.max_eval_samples is not None:
                eval_dataset = eval_dataset.select(range(self.args.max_eval_samples))

            eval_dataset = eval_dataset.map(
                self.preprocess_function,
                batched=True,
                num_proc=self.args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not self.args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
            eval_dataset.set_transform(self.transform_func)

            self.eval_dataset = eval_dataset

        if do_predict:
            self.max_target_length = self.args.val_max_target_length
            predict_dataset = raw_datasets["test"]
            if self.args.max_predict_samples is not None:
                predict_dataset = predict_dataset.select(
                    range(self.args.max_predict_samples)
                )
            
            predict_dataset = predict_dataset.map(
                self.preprocess_function,
                batched=True,
                num_proc=self.args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not self.args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
            predict_dataset.set_transform(self.transform_func)

            self.predict_dataset = predict_dataset

    def get_data_collator(self, model, fp16=True):
        # Data collator
        return DataCollatorForSeq2SeqWithImages(
            self.tokenizer,
            model=model,
            pad_to_multiple_of=8 if fp16 else None,
            args=self.args,
        )

    def transform_func(self, batch):
        images = batch.pop("images")[0]
        image_embeds = [
            torch.from_numpy(np.load(os.path.join(self.args.image_dir, image + ".npy"))).unsqueeze(0)
            #[0,:].unsqueeze(0)
            for image in images[:self.args.max_images]
        ]
        batch["image_embeds"] = torch.cat(image_embeds).unsqueeze(0)

        if "image_label" in batch:
            # convert indices to labels
            image_label = batch["image_label"][0]

            lab = [0] * len(image_embeds)
            for l in image_label:
                if l < len(image_embeds):
                    lab[l] = 1
            batch["image_label"] = [lab]

        return batch

    def preprocess_function(self, examples):
        # remove pairs where at least one record is None

        inputs, targets = [], []
        for i in range(len(examples[self.args.text_column])):
            if (
                examples[self.args.text_column][i] is not None
                and examples[self.args.summary_column][i] is not None
            ):
                inputs.append(examples[self.args.text_column][i])
                targets.append(examples[self.args.summary_column][i])

        inputs = examples[self.args.text_column]
        targets = examples[self.args.summary_column]
        # inputs = [prefix + inp for inp in inputs]
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.args.max_source_length - self.args.max_images,
            padding=self.padding,
            truncation=True,
        )

        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                targets,
                max_length=self.max_target_length,
                padding=self.padding,
                truncation=True,
            )

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if self.padding == "max_length" and self.args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label]
                for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        # model_inputs["decoder_mask"] = labels["attention_mask"]
        return model_inputs
