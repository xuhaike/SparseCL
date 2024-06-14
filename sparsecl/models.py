import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import transformers
from transformers import RobertaTokenizer
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead, BertEncoder, BertLayer, BertEmbeddings, BertPooler
from transformers.activations import gelu
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions, BaseModelOutputWithPastAndCrossAttentions
import math
from torch.utils.checkpoint import checkpoint
from typing import List, Optional, Tuple, Union
import logging
logger = logging.getLogger(__name__)

class OurBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = config.gradient_checkpointing if hasattr(config, 'gradient_checkpointing') else False

    def _gradient_checkpointing_func(self, layer_call, *args, **kwargs):
        """
        A wrapper function for performing gradient checkpointing on a single layer.
        The `layer_call` is a reference to the layer's `__call__` method.
        """
        # `checkpoint` function only accepts tensors as inputs, so filter args to remove None or other non-tensor objects
        tensor_args = tuple(arg for arg in args if isinstance(arg, torch.Tensor))
        # Use the `checkpoint` function from PyTorch, pass the callable (layer_call), followed by the arguments it needs
        return checkpoint(layer_call, *tensor_args, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        next_decoder_cache = () if use_cache else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            layer_args = (
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module, *layer_args
                )
            else:
                layer_outputs = layer_module(*layer_args)

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class OurBertModel(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = OurBertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[1]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError

def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = cls.model_args.pooler_type
    cls.pooler = Pooler(cls.model_args.pooler_type)
    if cls.model_args.pooler_type == "cls":
        cls.mlp = MLPLayer(config)
    cls.sim = Similarity(temp=cls.model_args.temp)

    cls.init_weights()

    cls.hidden_size=config.hidden_size
    cls.sqrt_hidden_size=math.sqrt(cls.hidden_size)
    cls.current_training_progress=0
    print(cls.hidden_size,cls.sqrt_hidden_size)

def our_cl_forward(cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    mlm_input_ids=None,
    mlm_labels=None,
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    ori_input_ids = input_ids
    batch_size = input_ids.size(0)
    # Number of sentences in one instance
    # 2: pair instance; 3: pair instance with a hard negative
    num_sent = input_ids.size(1)

    mlm_outputs = None
    # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
    attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent len)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)

    # Get raw embeddings
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    # MLM auxiliary objective
    if mlm_input_ids is not None:
        mlm_input_ids = mlm_input_ids.view((-1, mlm_input_ids.size(-1)))
        mlm_outputs = encoder(
            mlm_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,
        )

    # Pooling
    pooler_output = cls.pooler(attention_mask, outputs)
    pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1))) # (bs, num_sent, hidden)

    # If using "cls", we add an extra MLP layer
    # (same as BERT's original implementation) over the representation.
    # if cls.pooler_type == "cls":
        # pooler_output = cls.mlp(pooler_output)

    # Separate representation
    z1, z2 = pooler_output[:,0], pooler_output[:,1]

    # Hard negative
    if num_sent >2:
        z3 = pooler_output[:, 2]

    if num_sent >3:
        z4 = pooler_output[:, 3]

    # Gather all embeddings if using distributed training
    if dist.is_initialized() and cls.training:
        # Gather hard negative
        if num_sent >= 3:
            z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
            z3_list[dist.get_rank()] = z3
            z3 = torch.cat(z3_list, 0)

        # Dummy vectors for allgather
        z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
        z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
        # Allgather
        dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
        dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

        # Since allgather results do not have gradients, we replace the
        # current process's corresponding embeddings with original tensors
        z1_list[dist.get_rank()] = z1
        z2_list[dist.get_rank()] = z2
        # Get full batch embeddings: (bs x N, hidden)
        z1 = torch.cat(z1_list, 0)
        z2 = torch.cat(z2_list, 0)

    loss_type=cls.model_args.loss_type

    # print(loss_type)

    if loss_type=="cos":
        cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
        # Hard negative
        if num_sent >= 3:
            z1_z3_cos = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0))
            cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)

        labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
        loss_fct = nn.CrossEntropyLoss()

        # Calculate loss with hard negatives
        if num_sent == 3:
            # Note that weights are actually logits of weights
            z3_weight = cls.model_args.hard_negative_weight
            weights = torch.tensor(
                [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
            ).to(cls.device)
            cos_sim = cos_sim + weights

        loss = loss_fct(cos_sim, labels)
        cls.custom_epoch_info["cl_loss"].append(loss)
    elif loss_type=="sparsity":
        normalize_z1=z1/torch.norm(z1, dim=1, keepdim=True)
        normalize_z2=z2/torch.norm(z2, dim=1, keepdim=True)
        normalize_z3=z3/torch.norm(z3, dim=1, keepdim=True)

        labels = torch.arange(z1.size(0)).long().to(cls.device)
        loss_fct = nn.CrossEntropyLoss()

        normalized_z12=normalize_z1-normalize_z2
        normalized_z13=normalize_z1-normalize_z3
        normalized_z12_all=normalize_z1.unsqueeze(1)-normalize_z2.unsqueeze(0)
        normalized_z13_all=normalize_z1.unsqueeze(1)-normalize_z3.unsqueeze(0)


        l1_normalized_z12=torch.clamp(torch.norm(normalized_z12,p=1,dim=-1),min=cls.sqrt_hidden_size*1e-6)
        l2_normalized_z12=torch.clamp(torch.norm(normalized_z12,p=2,dim=-1),min=1e-6)
        l1l2_ratio_z12=torch.mean(l1_normalized_z12/l2_normalized_z12)

        l1_normalized_z13=torch.clamp(torch.norm(normalized_z13,p=1,dim=-1),min=cls.sqrt_hidden_size*1e-6)
        l2_normalized_z13=torch.clamp(torch.norm(normalized_z13,p=2,dim=-1),min=1e-6)
        l1l2_ratio_z13=torch.mean(l1_normalized_z13/l2_normalized_z13)


        l1_normalized_z12_all=torch.clamp(torch.norm(normalized_z12_all,p=1,dim=-1),min=cls.sqrt_hidden_size*1e-6)
        l2_normalized_z12_all=torch.clamp(torch.norm(normalized_z12_all,p=2,dim=-1),min=1e-6)
        l1l2_ratio_z12_all=torch.mean(l1_normalized_z12_all/l2_normalized_z12_all)
        hoyer_z12=(cls.sqrt_hidden_size-l1_normalized_z12_all/l2_normalized_z12_all)/(cls.sqrt_hidden_size-1)/cls.model_args.temp

        l1_normalized_z13_all=torch.clamp(torch.norm(normalized_z13_all,p=1,dim=-1),min=cls.sqrt_hidden_size*1e-6)
        l2_normalized_z13_all=torch.clamp(torch.norm(normalized_z13_all,p=2,dim=-1),min=1e-6)
        l1l2_ratio_z13_all=torch.mean(l1_normalized_z13_all/l2_normalized_z13_all)

        weights = torch.tensor([[0.0] * i + [cls.model_args.hard_negative_weight] + [0.0] * (z3.size(0) - i - 1) for i in range(z3.size(0))]).to(cls.device)

        hoyer_z13=(cls.sqrt_hidden_size+weights-l1_normalized_z13_all/l2_normalized_z13_all)/(cls.sqrt_hidden_size-1)/cls.model_args.temp

        hoyer_sim=torch.cat([hoyer_z12,hoyer_z13],1)

        cos_sim=hoyer_sim

        sparsity_loss=loss_fct(hoyer_sim,labels)

        loss=sparsity_loss

        # cls.custom_epoch_info["cl_loss"].append(cl_loss)
        cls.custom_epoch_info["sparsity_loss"].append(sparsity_loss)
        cls.custom_epoch_info["l1l2_ratio_z12"].append(l1l2_ratio_z12)
        cls.custom_epoch_info["l1l2_ratio_z13"].append(l1l2_ratio_z13)
        cls.custom_epoch_info["l1l2_ratio_z13_all"].append(l1l2_ratio_z13_all)
    elif loss_type=="sparsity-l2l1":
        normalize_z1=z1/torch.norm(z1, dim=1, keepdim=True)
        normalize_z2=z2/torch.norm(z2, dim=1, keepdim=True)
        normalize_z3=z3/torch.norm(z3, dim=1, keepdim=True)

        labels = torch.arange(z1.size(0)).long().to(cls.device)
        loss_fct = nn.CrossEntropyLoss()

        normalized_z12=normalize_z1-normalize_z2
        normalized_z13=normalize_z1-normalize_z3
        normalized_z12_all=normalize_z1.unsqueeze(1)-normalize_z2.unsqueeze(0)
        normalized_z13_all=normalize_z1.unsqueeze(1)-normalize_z3.unsqueeze(0)


        l1_normalized_z12=torch.clamp(torch.norm(normalized_z12,p=1,dim=-1),min=1e-6)
        l2_normalized_z12=torch.norm(normalized_z12,p=2,dim=-1)
        l2l1_ratio_z12=torch.mean(l2_normalized_z12/l1_normalized_z12)

        l1_normalized_z13=torch.clamp(torch.norm(normalized_z13,p=1,dim=-1),min=1e-6)
        l2_normalized_z13=torch.norm(normalized_z13,p=2,dim=-1)
        l2l1_ratio_z13=torch.mean(l2_normalized_z13/l1_normalized_z13)


        l1_normalized_z12_all=torch.clamp(torch.norm(normalized_z12_all,p=1,dim=-1),min=1e-6)
        l2_normalized_z12_all=torch.norm(normalized_z12_all,p=2,dim=-1)
        l2l1_ratio_z12_all=torch.mean(l2_normalized_z12_all/l1_normalized_z12_all)
        l2l1_z12=l2_normalized_z12_all/l1_normalized_z12_all/cls.model_args.temp

        l1_normalized_z13_all=torch.clamp(torch.norm(normalized_z13_all,p=1,dim=-1),min=1e-6)
        l2_normalized_z13_all=torch.norm(normalized_z13_all,p=2,dim=-1)
        l2l1_ratio_z13_all=torch.mean(l2_normalized_z13_all/l1_normalized_z13_all)

        weights = torch.tensor([[0.0] * i + [cls.model_args.hard_negative_weight] + [0.0] * (z3.size(0) - i - 1) for i in range(z3.size(0))]).to(cls.device)

        l2l1_z13=l2_normalized_z13_all/l1_normalized_z13_all/cls.model_args.temp

        l2l1_sim=torch.cat([l2l1_z12,l2l1_z13],1)

        cos_sim=None

        sparsity_loss=loss_fct(l2l1_sim,labels)

        loss=sparsity_loss

        # cls.custom_epoch_info["cl_loss"].append(cl_loss)
        cls.custom_epoch_info["sparsity_loss"].append(sparsity_loss)
        cls.custom_epoch_info["l2l1_ratio_z12"].append(l2l1_ratio_z12)
        cls.custom_epoch_info["l2l1_ratio_z13"].append(l2l1_ratio_z13)
        cls.custom_epoch_info["l2l1_ratio_z13_all"].append(l2l1_ratio_z13_all)
    elif loss_type=="sparsity-l422":
        normalize_z1=z1/torch.norm(z1, dim=1, keepdim=True)
        normalize_z2=z2/torch.norm(z2, dim=1, keepdim=True)
        normalize_z3=z3/torch.norm(z3, dim=1, keepdim=True)

        labels = torch.arange(z1.size(0)).long().to(cls.device)
        loss_fct = nn.CrossEntropyLoss()

        normalized_z12=normalize_z1-normalize_z2
        normalized_z13=normalize_z1-normalize_z3
        normalized_z12_all=normalize_z1.unsqueeze(1)-normalize_z2.unsqueeze(0)
        normalized_z13_all=normalize_z1.unsqueeze(1)-normalize_z3.unsqueeze(0)


        l4_normalized_z12=torch.sum(torch.pow(normalized_z12,4),dim=-1)
        l22_normalized_z12=torch.clamp(torch.pow(torch.sum(torch.pow(normalized_z12,2),dim=-1),2),min=1e-6)
        l422_ratio_z12=torch.mean(l4_normalized_z12/l22_normalized_z12)

        l4_normalized_z13=torch.sum(torch.pow(normalized_z13,4),dim=-1)
        l22_normalized_z13=torch.clamp(torch.pow(torch.sum(torch.pow(normalized_z13,2),dim=-1),2),min=1e-6)
        l422_ratio_z13=torch.mean(l4_normalized_z13/l22_normalized_z13)


        l4_normalized_z12_all=torch.sum(torch.pow(normalized_z12_all,4),dim=-1)
        l22_normalized_z12_all=torch.clamp(torch.pow(torch.sum(torch.pow(normalized_z12_all,2),dim=-1),2),min=1e-6)
        l422_ratio_z12_all=torch.mean(l4_normalized_z12_all/l22_normalized_z12)
        l422_z12=l4_normalized_z12_all/l22_normalized_z12_all/cls.model_args.temp

        l4_normalized_z13_all=torch.sum(torch.pow(normalized_z13_all,4),dim=-1)
        l22_normalized_z13_all=torch.clamp(torch.pow(torch.sum(torch.pow(normalized_z13_all,2),dim=-1),2),min=1e-6)
        l422_ratio_z13_all=torch.mean(l4_normalized_z13_all/l22_normalized_z13)
        l422_z13=l4_normalized_z13_all/l22_normalized_z13_all/cls.model_args.temp

        weights = torch.tensor([[0.0] * i + [cls.model_args.hard_negative_weight] + [0.0] * (z3.size(0) - i - 1) for i in range(z3.size(0))]).to(cls.device)

        l422_sim=torch.cat([l422_z12,l422_z13],1)

        cos_sim=None

        sparsity_loss=loss_fct(l422_sim,labels)

        loss=sparsity_loss

        # cls.custom_epoch_info["cl_loss"].append(cl_loss)
        cls.custom_epoch_info["sparsity_loss"].append(sparsity_loss)
        cls.custom_epoch_info["l422_ratio_z12"].append(l422_ratio_z12)
        cls.custom_epoch_info["l422_ratio_z13"].append(l422_ratio_z13)
        cls.custom_epoch_info["l422_ratio_z13_all"].append(l422_ratio_z13_all)

    # Calculate loss for MLM
    if mlm_outputs is not None and mlm_labels is not None:
        mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
        prediction_scores = cls.lm_head(mlm_outputs.last_hidden_state)
        masked_lm_loss = loss_fct(prediction_scores.view(-1, cls.config.vocab_size), mlm_labels.view(-1))
        loss = loss + cls.model_args.mlm_weight * masked_lm_loss

    if not return_dict:
        output = (cos_sim,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
    return SequenceClassifierOutput(
        loss=loss,
        logits=cos_sim,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

def sentemb_forward(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    pooler_output = cls.pooler(attention_mask, outputs)
    # if cls.pooler_type == "cls" and not cls.model_args.mlp_only_train:
        # pooler_output = cls.mlp(pooler_output)

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )


class our_BertForCL(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.bert = BertModel(config, add_pooling_layer=False)

        if self.model_args.do_mlm:
            self.lm_head = BertLMPredictionHead(config)
        self.custom_epoch_info={    "loss":[],"cl_loss": [], "sparsity_loss":[],
                                    "l1l2_ratio_z12":[],"l1l2_ratio_z13":[],"l1l2_ratio_z13_all":[],
                                    "l2l1_ratio_z12":[],"l2l1_ratio_z13":[],"l2l1_ratio_z13_all":[],
                                    "l422_ratio_z12":[],"l422_ratio_z13":[],"l422_ratio_z13_all":[],
                                }
        # self.custom_epoch_info = {"cl_loss": [], "sparsity_loss": [], "l1l2_ratio_z12": [], "l1l2_ratio_z13": [], "l1l2_ratio_z13_all": []}

        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return our_cl_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )
