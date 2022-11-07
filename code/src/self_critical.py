import torch
import torch.nn as nn
import torch.nn.functional as F

import clip
from bert_score import BERTScorer

from transformers.generation_stopping_criteria import StoppingCriteriaList, MaxLengthCriteria
from transformers import AutoConfig, AutoTokenizer

from rouge_score import rouge_scorer

class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, mask, reward, reduction='mean'):
        N,L = input.shape[:2]
        input = input.gather(2, seq.unsqueeze(2)).squeeze(2)
        
        input = input.reshape(-1)
        reward = reward.unsqueeze(1).expand(-1, L).reshape(-1)
        # reward = reward.reshape(-1)
        # mask = (seq>0).to(input)
        # mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1).reshape(-1)
        mask = mask.reshape(-1)

        output = - input * reward * mask
        
        if reduction == 'none':
            output = output.view(N,L).sum(1) / mask.view(N,L).sum(1)
        elif reduction == 'mean':
            output = torch.sum(output) / torch.sum(mask)

        return output

class SCSTTrainer:
    def __init__(self, reward_model, rouge_key=None, cbs_weight=1.0, rouge_weight=1.0, device="cuda"):
        print("loading modules for reward...")
        print("loading CLIP")
        # self.clip_model, _ = clip.load("ViT-B/32", device=device)
        self.clip_model, _ = clip.load("RN50x64", device=device)
        print("CLIP loaded")
        print("loading bertscore")
        self.bertscore = BERTScorer( model_type="roberta-large-mnli", num_layers=10, device=device)
        print("bertscore_loaded")

        self.rouge_key = rouge_key
        self.rouge = None
        if self.rouge_key is not None:
            print("loading ROUGE {}".format(self.rouge_key))
            self.rouge = rouge_scorer.RougeScorer([self.rouge_key], use_stemmer=True)

        self.cbs_weight = cbs_weight
        self.rouge_weight = rouge_weight

        self.device = device
        self.loss_func = RewardCriterion()

    def compute_self_critical_loss(self, model, tokenizer, inputs, i=None):

        # get necassary inputs
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        visn_features = inputs["visn_features"]
        # image_attention_mask = inputs["image_attention_mask"]
        
        unwrapped_model = model
        if hasattr(model, "module"):
            unwrapped_model = model.module
        max_length = unwrapped_model.config.max_length

        # greedy decoding
        with torch.no_grad():
            unwrapped_model.eval()
            greedy_output = unwrapped_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                visn_features=visn_features,
                # image_attention_mask=image_attention_mask,
            )
        
        # sample, hacky way to remove no_grad decorator
        # https://github.com/huggingface/transformers/issues/15552#issuecomment-1033154753

        unwrapped_model.train()
        sample_output = self.sample(
            unwrapped_model,
            inputs=input_ids,
            attention_mask=attention_mask,
            visn_features=visn_features,
            # image_attention_mask=image_attention_mask,
        )
        # remove decoder_start_token_id from output_sequences
        sample_sequences = sample_output.sequences[:, 1:]
        #sample_attention_mask = torch.where(sample_sequences!=tokenizer.pad_token_id, 1 , 0)
        sample_attention_mask = (sample_sequences!=tokenizer.pad_token_id).float()

        lm_logits_sampling = sample_output.scores
        lm_logits_sampling = torch.stack(lm_logits_sampling).transpose(0,1) # stack on tokens but needs to be at 1

        logprobs = F.log_softmax(lm_logits_sampling, dim=-1)

        # prepare sequence
        greedy_seq = tokenizer.batch_decode(greedy_output, skip_special_tokens=True,  clean_up_tokenization_spaces=False)
        sample_seq = tokenizer.batch_decode(sample_sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        documents = tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True,  clean_up_tokenization_spaces=False)
        
        references = inputs["labels"].clone()
        references[references == -100] = tokenizer.pad_token_id
        references = tokenizer.batch_decode(references, skip_special_tokens=True,  clean_up_tokenization_spaces=False )
        
        # calculate reward
        with torch.no_grad():
            image_embeds_for_reward = visn_features
            image_attention_mask_for_reward = None
            
            reward = self.get_self_critical_reward(
                documents,
                sample_seq,
                greedy_seq,
                image_embeds_for_reward,
                references,
                image_attention_mask_for_reward,
                i=i,
            )
            #_reward = reward["reward"].to(best_logprobs_sampling.device).float()
            _reward = reward["reward"].to(logprobs.device).float()
     
        # sc_loss = self.loss_func(best_logprobs_sampling, sample_attention_mask, _reward)
        sc_loss = self.loss_func(logprobs, sample_sequences, sample_attention_mask, _reward)
        return sc_loss, reward

    def get_self_critical_reward(self, document, sample_output, greedy_output, image_embeds, references, image_attention_mask=None, i=None):
        """
        greedy_res: result using greedy decoding
        data_gts: ground truth sequence
        gen_result: generated result
        """
        # combine sample and greedy to save computation

        # document and image_features are of size batch_size, while summary is both sampled and greedy
        rewards = self.compute_reward(
            document + document,
            sample_output + greedy_output,
            image_embeds,
            references + references,
            image_attention_mask,
            i=i,
        )
        N = len(document)
        
        sample_reward, greedy_reward = rewards[:N], rewards[N:]
        
        reward = sample_reward - greedy_reward
        return {"reward": reward, "sample_reward": sample_reward, "greedy_reward": greedy_reward}
    
    def compute_reward(self, document, summary, image_features, reference, image_attention_mask=None, i=None):
        """
        source: str = document
        image: Tensor = CLIP image features
        target: str = summary
        """

        if i % 2 == 0  or self.rouge is None:
            P, _, _ = self.bertscore.score(summary, document)
            P = P.to(image_features.device)

            text = clip.tokenize(summary, truncate=True).to(image_features.device)
            text_features = self.clip_model.encode_text(text)

            # normalized features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # double image features, since we have both greedy and sampled summaries
            image_features = torch.cat([image_features, image_features], dim=0)
            B, N, _ = image_features.size()

            image_features = image_features.view(B*N, -1)
            logit_scale = self.clip_model.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_image = logits_per_image.view(B, N, -1).nan_to_num()

            if image_attention_mask is not None: # we only take the mean across valid ones
                image_attention_mask = torch.cat([image_attention_mask, image_attention_mask], dim=0) # since we doubled for greedy and sample
                image_mask_extended = image_attention_mask.unsqueeze(-1).expand(logits_per_image.size()).float()
                logits_per_image = torch.sum(logits_per_image * image_mask_extended, 1) / torch.clamp(image_mask_extended.sum(1), min=1e-9)
            else:
                logits_per_image = logits_per_image.mean(dim=1)
            
            clip_score = logits_per_image[torch.eye(logits_per_image.size(0), dtype=torch.bool)]
            scores = torch.add(P*0.75, clip_score/100*0.25) * self.cbs_weight
        else:
            rouge = []
            for summ, ref in zip(summary, reference):
                rouge.append(self.rouge.score(ref, summ)[self.rouge_key].fmeasure)
            scores = torch.tensor(rouge, device=image_features.device) * self.rouge_weight
        return scores
    
    def sample(self, model, inputs, max_length=None, **model_kwargs):
        bos_token_id = model.config.bos_token_id
        pad_token_id = model.config.pad_token_id
        eos_token_id = model.config.eos_token_id

        # _prepare_model_inputs ()
        if (
            model.config.is_encoder_decoder
            and hasattr(model, "encoder")
            and model.encoder.main_input_name != model.main_input_name
        ):
            input_name = model.encoder.main_input_name
        else:
            input_name = model.main_input_name

        model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None or k != input_name}

        # 1.2. check whether model_input_name is passed as kwarg
        # if yes and `inputs` is None use kwarg inputs
        inputs_kwarg = model_kwargs.pop(input_name, None)
        if inputs_kwarg is not None and inputs is not None:
            raise ValueError(
                f"`inputs`: {inputs}` were passed alongside "
                f"{input_name} which is not allowed."
                f"Make sure to either pass {inputs} or {input_name}=..."
            )
        elif inputs_kwarg is not None:
            inputs = inputs_kwarg

        # 1.3. models with `input_ids` can also make use of `inputs_embeds`
        if model._can_retrieve_inputs_from_name(inputs, "inputs_embeds", model_kwargs):
            inputs, input_name = model_kwargs["inputs_embeds"], "inputs_embeds"

        # 1.4. Only encoder-decoder models can have non `input_ids` input format
        if not model.config.is_encoder_decoder and input_name != "input_ids":
            raise ValueError(
                f"If {input_name} is passed as model-specific keyword "
                "input then model has to be an encoder-decoder and not a "
                f"{model.__class__.__name__}."
            )

        # 1.5. if `inputs` is still None, try to create `input_ids` from BOS token
        if inputs is None:
            inputs = model._prepare_input_ids_for_generation(bos_token_id, model_kwargs.get("encoder_outputs"))
        batch_size = inputs.shape[0]

        # 2. Define other model kwargs
        if model.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created
            # and added to `model_kwargs`
            model_kwargs = model._prepare_encoder_decoder_kwargs_for_generation(
                inputs, model_kwargs, input_name
            )

        # 4. Prepare `input_ids` which will be used for auto-regressive generation
        if model.config.is_encoder_decoder:
            input_ids = model._prepare_decoder_input_ids_for_generation(
                batch_size,
                decoder_start_token_id=None,
                bos_token_id=bos_token_id,
                model_kwargs=model_kwargs,
            )
        else:
            # if decoder-only then inputs_tensor has to be `input_ids`
            input_ids = inputs

        input_ids_seq_length = input_ids.shape[-1]

        # 5. Prepare `max_length` depending on other stopping criteria
        # default to config if still None
        max_length = max_length if max_length is not None else model.config.max_length

        if input_ids_seq_length >= max_length:
            input_ids_string = "decoder_input_ids" if model.config.is_encoder_decoder else "input_ids"
            logger.warning(
                f"Input length of {input_ids_string} is {input_ids_seq_length}, but ``max_length`` is set to {max_length}. "
                "This can lead to unexpected behavior. You should consider increasing ``config.max_length`` or ``max_length``."
            )
        
        # 8. prepare stopping criteria
        stopping_criteria = StoppingCriteriaList()
        if max_length is not None:
            stopping_criteria.append(MaxLengthCriteria(max_length=max_length))
        
        
        # actual sampling method
        return model.sample(
            input_ids,
            logits_processor=None,
            logits_warper=None,
            stopping_criteria=stopping_criteria,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            output_scores=True,
            return_dict_in_generate=True,
            synced_gpus=False,
            **model_kwargs,
        )
    
    def bertscore_combine_max(
        self,
        doc_emb,
        img_emb,
        summ_emb,
        doc_img_idf,
        summ_idf,
        doc_summ_masks,
        img_summ_masks,
        is_scale=1.0
    ):
        ds_sim = torch.bmm(summ_emb, doc_emb.transpose(1, 2))
        ds_sim = ds_sim * doc_summ_masks.float()
        is_sim = torch.bmm(summ_emb, img_emb.transpose(1, 2))
        is_sim = is_sim * img_summ_masks.float()
        is_sim = is_scale * is_sim

        ds_word_precision = ds_sim.max(dim=2)[0]
        ds_word_recall = ds_sim.max(dim=1)[0]
        
        is_word_precision = is_sim.max(dim=2)[0]
        is_word_recall = is_sim.max(dim=1)[0]

        word_precision = torch.cat([ds_word_precision, is_word_precision], dim=1)
        word_recall = torch.cat([ds_word_recall, is_word_recall], dim=1)

        P = (word_precision * summ_idf).sum(1)
        R = (word_recall * doc_img_idf).sum(dim=1)
        F = 2 * P * R / (P+R)
        return P, R, F
