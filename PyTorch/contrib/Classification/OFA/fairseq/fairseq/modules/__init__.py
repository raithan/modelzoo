"""isort:skip_file"""
import math
import torch

# 在文件开头的导入部分添加这些导入
from .adaptive_input import AdaptiveInput
from .adaptive_softmax import AdaptiveSoftmax
from .base_layer import BaseLayer
from .beamable_mm import BeamableMM
from .character_token_embedder import CharacterTokenEmbedder
from .conv_tbc import ConvTBC
from .cross_entropy import cross_entropy
from .downsampled_multihead_attention import DownsampledMultiHeadAttention
from .dynamic_convolution import DynamicConv, DynamicConv1dTBC
from .dynamic_crf_layer import DynamicCRF
from .fairseq_dropout import FairseqDropout
from .fp32_group_norm import Fp32GroupNorm
from .gelu import gelu, gelu_accurate
from .grad_multiply import GradMultiply
from .gumbel_vector_quantizer import GumbelVectorQuantizer
from .kmeans_vector_quantizer import KmeansVectorQuantizer
from .layer_drop import LayerDropModuleList
from .layer_norm import Fp32LayerNorm, LayerNorm
from .learned_positional_embedding import LearnedPositionalEmbedding
from .lightweight_convolution import LightweightConv, LightweightConv1dTBC
from .linearized_convolution import LinearizedConvolution
from .location_attention import LocationAttention
from .lstm_cell_with_zoneout import LSTMCellWithZoneOut
from .multihead_attention import MultiheadAttention
# from .positional_embedding import PositionalEmbedding, RelPositionalEncoding
from .positional_embedding import PositionalEmbedding
from .same_pad import SamePad
from .scalar_bias import ScalarBias
from .sinusoidal_positional_embedding import SinusoidalPositionalEmbedding
from .transformer_sentence_encoder_layer import TransformerSentenceEncoderLayer
from .transformer_sentence_encoder import TransformerSentenceEncoder
from .transpose_last import TransposeLast
from .unfold import unfold1d
from .transformer_layer import TransformerDecoderLayer, TransformerEncoderLayer
from .vggblock import VGGBlock
from .espnet_multihead_attention import (
    ESPNETMultiHeadedAttention,
    RelPositionMultiHeadedAttention,
    RotaryPositionMultiHeadedAttention,
)
from .rotary_positional_embedding import RotaryPositionalEmbedding

__all__ = [
    "AdaptiveInput",
    "AdaptiveSoftmax",
    "BaseLayer",
    "BeamableMM",
    "CharacterTokenEmbedder",
    "ConvTBC",
    "cross_entropy",
    "DownsampledMultiHeadAttention",
    "DynamicConv1dTBC",
    "DynamicConv",
    "DynamicCRF",
    "ESPNETMultiHeadedAttention",
    "FairseqDropout",
    "Fp32GroupNorm",
    "Fp32LayerNorm",
    "gelu",
    "gelu_accurate",
    "GradMultiply",
    "GumbelVectorQuantizer",
    "KmeansVectorQuantizer",
    "LayerDropModuleList",
    "LayerNorm",
    "LearnedPositionalEmbedding",
    "LightweightConv1dTBC",
    "LightweightConv",
    "LinearizedConvolution",
    "LocationAttention",
    "LSTMCellWithZoneOut",
    "MultiheadAttention",
    "PositionalEmbedding",
    # "RelPositionalEncoding",
    "RelPositionMultiHeadedAttention",
    "RotaryPositionMultiHeadedAttention",
    "RotaryPositionalEmbedding",
    "SamePad",
    "ScalarBias",
    "SinusoidalPositionalEmbedding",
    "TransformerSentenceEncoderLayer",
    "TransformerSentenceEncoder",
    "TransformerDecoderLayer",
    "TransformerEncoderLayer",
    "TransposeLast",
    "VGGBlock",
    "unfold1d",
]
# RelPositionalEncoding 占位符实现
class RelPositionalEncoding:
    """Placeholder implementation for RelPositionalEncoding"""
    def __init__(self, max_positions, embedding_dim):
        self.max_positions = max_positions
        self.embedding_dim = embedding_dim
        # 创建一个简单的正弦位置编码作为替代
        self.pe = None
        
    def forward(self, x):
        # 如果还没有创建位置编码，创建一个简单的版本
        if self.pe is None:
            positions = torch.arange(x.size(1), device=x.device).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, self.embedding_dim, 2, device=x.device)
                * -(math.log(10000.0) / self.embedding_dim)
            )
            pe = torch.zeros(x.size(1), self.embedding_dim, device=x.device)
            pe[:, 0::2] = torch.sin(positions * div_term)
            pe[:, 1::2] = torch.cos(positions * div_term)
            self.pe = pe.unsqueeze(0)
        
        # 返回输入加上位置编码
        return x + self.pe[:, :x.size(1)]