from logging import raiseExceptions
import random
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import os
import torch
import numpy as np

from transformers import PreTrainedTokenizerBase


@dataclass
class DataCollatorForSeq2SeqWithImages:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str] = True
    pad_to_multiple_of: Optional[int] = None
    args: Optional[Any] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Largely adapted from DataCollatorForSeq2Seq, but also transforms images into the correct format
        """

        # Deal with images
        image_embeds = [feature["image_embeds"] for feature in features]
        max_image_length = self.args.max_images

        # pad image features and create mask
        # take image_embeds out of features for not since we do not want to feed it to pad()
        image_embeds = []
        for feature in features:
            image_embed = feature.pop("image_embeds")
            image_attention_mask = [1] * len(image_embed)
            n_remaining = max_image_length - len(image_embed)
            remainder = torch.zeros(n_remaining, self.args.visual_feat_dim)
            image_embed = torch.cat([image_embed, remainder])
            image_attention_mask += [0] * n_remaining
            feature["image_attention_mask"] = image_attention_mask
            image_embeds.append(image_embed.unsqueeze(0))
        image_embeds = torch.cat(image_embeds)

        import numpy as np

        labels = (
            [feature["labels"] for feature in features]
            if "labels" in features[0].keys()
            else None
        )
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            label_pad_token_id = -100
            for feature in features:
                remainder = [label_pad_token_id] * (
                    max_label_length - len(feature["labels"])
                )
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder
                        if padding_side == "right"
                        else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate(
                        [feature["labels"], remainder]
                    ).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate(
                        [remainder, feature["labels"]]
                    ).astype(np.int64)

        # pad image labels
        image_labels = (
            [feature["image_label"] for feature in features]
            if "image_label" in features[0].keys()
            else None
        )
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if image_labels is not None:
            max_label_length = self.args.max_images
            # max_label_length = max(len(l) for l in image_labels)
            # if self.pad_to_multiple_of is not None:
            #     max_label_length = (
            #         (max_label_length + self.pad_to_multiple_of - 1)
            #         // self.pad_to_multiple_of
            #         * self.pad_to_multiple_of
            #     )

            padding_side = self.tokenizer.padding_side
            label_pad_token_id = -100
            for feature in features:
                remainder = [label_pad_token_id] * (
                    max_label_length - len(feature["image_label"])
                )
                feature["image_label"] = (
                    feature["image_label"] + remainder
                    if padding_side == "right"
                    else remainder + feature["image_label"]
                )

        max_length = self.args.max_source_length - self.args.max_images
        # if self.args.max_source_length + 50*self.args.max_images > self.model.config.max_position_embeddings:
        #    max_length -= (self.model.config.max_position_embeddings - (self.args.max_source_length + max_image_length) )
        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        attention_mask = features["attention_mask"]
        image_attention_mask = features.pop("image_attention_mask")
        
        features["attention_mask"] = torch.cat([attention_mask, image_attention_mask], dim=1)
        features["visn_features"] = image_embeds


        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(
                labels=features["labels"]
            )
            features["decoder_input_ids"] = decoder_input_ids
        
        return features
