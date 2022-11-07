import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Seq2SeqTrainer

from self_critical import SCSTTrainer

class XMTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        # add image loss
        if outputs.visual_loss is not None:
            loss += 6*outputs.visual_loss
        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        """
        Perform an evaluation step on `model` using `inputs`.
        Added necassary inputs to the model
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model,
                inputs,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = {
            "max_length": self._max_length
            if self._max_length is not None
            else self.model.config.max_length,
            "num_beams": self._num_beams
            if self._num_beams is not None
            else self.model.config.num_beams,
            "synced_gpus": False,
            # "synced_gpus": True if is_deepspeed_zero3_enabled() else False,
        }

        # prepare generation inputs
        # some encoder-decoder models can have varying encder's and thus
        # varying model input names
        if (
            hasattr(self.model, "encoder")
            and self.model.encoder.main_input_name != self.model.main_input_name
        ):
            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[self.model.main_input_name]
        
        generated_tokens = self.model.generate(
            generation_inputs,
            attention_mask=inputs.get("attention_mask", None),
            visn_features=inputs["visn_features"],
            **gen_kwargs,
        )
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(
                generated_tokens, gen_kwargs["max_length"]
            )

        loss = None
        # with torch.no_grad():
        #     with self.autocast_smart_context_manager():
        #         outputs = model(**inputs)
        #     if has_labels:
        #         if self.label_smoother is not None:
        #             loss = (
        #                 self.label_smoother(outputs, inputs["labels"]).mean().detach()
        #             )
        #         else:
        #             loss = (
        #                 (outputs["loss"] if isinstance(outputs, dict) else outputs[0])
        #                 .mean()
        #                 .detach()
        #             )
        #     else:
        #         loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_kwargs["max_length"]:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
        else:
            labels = None

        # To output image recommendation use that
        # need to get decoder hidden state with respect to the generated summary
        inputs["decoder_input_ids"] = generated_tokens
        inputs.pop("labels")
        with torch.no_grad():
            with self.autocast_smart_context_manager():
                outputs = model(**inputs)

                # decoder_input_ids = generated_tokens
                # decoder_last_hidden_state = outputs.decoder_last_hidden_state
                
                # decoder_attention_mask = (decoder_input_ids == self.model.config.pad_token_id)
                # text_features = self.model.mean_pooling(decoder_last_hidden_state, decoder_attention_mask).to(dtype=self.model.dtype)
                # image_features = self.model.visual_proj(inputs["visn_features"])
                # visual_logits = torch.cat([image_features, text_features.unsqueeze(1).expand(-1,image_features.size(1),-1)], dim = -1)
                # visual_logits = self.model.visual_head(visual_logits)
        visual_logits = outputs.visual_logits
        if visual_logits is not None:
            return (loss, (generated_tokens, visual_logits), labels)

        return (loss, generated_tokens, labels)


class SelfCriticalTrainer(XMTrainer):
    def __init__(self, reward_model, config=None, data_args=None, rl_weight=0.998, rouge_key=None, cbs_weight=1.0, rouge_weight=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scst = SCSTTrainer(reward_model, rouge_key=rouge_key, cbs_weight=cbs_weight, rouge_weight=rouge_weight)
        self.i_step = 0
        self.rl_weight = rl_weight

    def compute_loss(self, model, inputs, return_outputs=False):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)

        rl_loss, rewards = self.scst.compute_self_critical_loss(
            model,
            self.tokenizer,
            inputs,
            i=self.i_step,
        )

        if self.i_step % 100 == 0:
            print("reward at {}".format(self.i_step), {k: v.mean().item() for k,v in rewards.items()})
            print("rl loss at {}".format(self.i_step), rl_loss)
        self.i_step += 1
        
        # combine loss
        loss = self.rl_weight*rl_loss + (1-self.rl_weight)*loss

        return (loss, outputs) if return_outputs else loss