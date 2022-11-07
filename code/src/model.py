import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from dataclasses import dataclass
from typing import Optional, Tuple

from transformers.models.bart.modeling_bart import (
    BartForConditionalGeneration,
    BartModel,
    BartDecoder,
    BartEncoder,
    BartAttention,
    shift_tokens_right,
    _expand_mask,
)

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    Seq2SeqModelOutput,
    Seq2SeqLMOutput,
    BaseModelOutput,
)

from transformers.models.lxmert.modeling_lxmert import LxmertLayer, LxmertXLayer

@dataclass
class XMBaseModelOutput(BaseModelOutput):
    last_hidden_state: torch.FloatTensor = None
    visual_last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    visual_attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class XMSeq2SeqModelOutput(Seq2SeqModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    visual_encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    visual_encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class XMSeq2SeqLMOutput(Seq2SeqLMOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_last_hidden_state: Optional[torch.FloatTensor] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    visual_logits: Optional[torch.FloatTensor] = None
    visual_loss: Optional[torch.FloatTensor] = None

class VLBartEncoder(BartEncoder):
    def __init__(self, config):
        super().__init__(config)
        
        self.visn_positions = nn.Embedding(config.max_images, config.d_model)
        self.visn_projection = nn.Linear(config.visual_feat_dim, config.d_model)

        self.post_init()
    
    def forward(
        self,
        input_ids=None,
        attention_mask = None,
        head_mask = None,
        inputs_embeds = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
        visn_features=None,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
            visn_features = self.embed_visn(visn_features)
            inputs_embeds = torch.cat([inputs_embeds, visn_features], dim=1)
            input_ids = None
        return super().forward(input_ids, attention_mask,  head_mask, inputs_embeds, output_attentions, output_hidden_states, return_dict)
    
    def embed_visn(self,visn_features):
        m, n, dim = visn_features.size()
        visn_features = visn_features.view(-1, dim)
        visn_features = self.visn_projection(visn_features)
        visn_features = visn_features.view(m, n, -1)
        
        visn_pos = self.visn_positions(torch.arange(n, dtype=torch.long, device=visn_features.device))
        return visn_features + visn_pos


class VLBartModel(BartModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = VLBartEncoder(config)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        visn_features=None,
    ):

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                visn_features=visn_features,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return XMSeq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_attentions=encoder_outputs.attentions,
            visual_encoder_last_hidden_state=None,
            visual_encoder_attentions=None,
        )

class XMCLIPBARTForConditionalGeneration(BartForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.model = VLBartModel(config)

        self.visual_dense = nn.Linear(config.d_model, config.d_model)
        self.visual_dropout = nn.Dropout(config.classifier_dropout)
        self.visual_head = nn.Linear(config.d_model, 2)
        # self.visual_proj = nn.Linear(config.visual_feat_dim, config.d_model)
        # self.visual_head = nn.Linear(config.d_model*2, 2, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        visn_features=None,
        image_label=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
        
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            visn_features=visn_features,
        )

        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
        
        visual_loss = None
        visual_logits=None

        visual_logits = outputs.encoder_last_hidden_state[:,-self.config.max_images:,:]
        visual_logits = self.visual_dropout(visual_logits)
        visual_logits = self.visual_dense(visual_logits)
        visual_logits = torch.tanh(visual_logits)
        visual_logits = self.visual_dropout(visual_logits)
        visual_logits = self.visual_head(visual_logits).reshape(-1,self.config.max_images,2)
        
        if image_label is not None:
            # decoder_attention_mask = (decoder_input_ids == self.config.pad_token_id)
            # text_features = self.mean_pooling(outputs.last_hidden_state, decoder_attention_mask).to(dtype=self.dtype)
            # image_features = self.visual_proj(visn_features)
            # visual_logits = torch.cat([image_features, text_features.unsqueeze(1).expand(-1,image_features.size(1),-1)], dim = -1)
            # visual_logits = self.visual_head(visual_logits)

            loss_fct = CrossEntropyLoss()
            visual_loss = loss_fct(visual_logits.reshape(-1, 2), image_label.view(-1))
        return XMSeq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_last_hidden_state=outputs.last_hidden_state,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            visual_loss=visual_loss,
            visual_logits=visual_logits,
        )

    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
