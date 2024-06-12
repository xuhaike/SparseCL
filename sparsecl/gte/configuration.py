# coding=utf-8
# Copyright 2024 The GTE Team Authors and Alibaba Group.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" NEW model configuration"""
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class NewConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`NewModel`] or a [`TFNewModel`]. It is used to
    instantiate a NEW model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the NEW
    [izhx/new-base-en](https://huggingface.co/izhx/new-base-en) architecture.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the NEW model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`NewModel`] or [`TFNewModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`NewModel`] or [`TFNewModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        position_embedding_type (`str`, *optional*, defaults to `"rope"`):
            Type of position embedding. Choose one of `"absolute"`, `"rope"`.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
            strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
            `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
            `max_position_embeddings` to the expected new maximum. See the following thread for more information on how
            these scaling strategies behave:
            https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/. This is an
            experimental feature, subject to breaking API changes in future versions.
        classifier_dropout (`float`, *optional*):
            The dropout ratio for the classification head.
    Examples:
    ```python
    >>> from transformers import NewConfig, NewModel
    >>> # Initializing a NEW izhx/new-base-en style configuration
    >>> configuration = NewConfig()
    >>> # Initializing a model (with random weights) from the izhx/new-base-en style configuration
    >>> model = NewModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "new"

    def __init__(
        self,
        vocab_size=30528,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.0,
        max_position_embeddings=2048,
        type_vocab_size=1,
        initializer_range=0.02,
        layer_norm_type='layer_norm',
        layer_norm_eps=1e-12,
        # pad_token_id=0,
        position_embedding_type="rope",
        rope_theta=10000.0,
        rope_scaling=None,
        classifier_dropout=None,
        pack_qkv=True,
        unpad_inputs=False,
        use_memory_efficient_attention=False,
        logn_attention_scale=False,
        logn_attention_clip1=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_type = layer_norm_type
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.classifier_dropout = classifier_dropout

        self.pack_qkv = pack_qkv
        self.unpad_inputs = unpad_inputs
        self.use_memory_efficient_attention = use_memory_efficient_attention
        self.logn_attention_scale = logn_attention_scale
        self.logn_attention_clip1 = logn_attention_clip1
