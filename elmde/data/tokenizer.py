"""
Production Tokenizer Integration for ELMDE.
Supports hardware-aligned 50,368-token vocabularies (ModernBERT / BERT)
with fast tokenization, jagged unpadded sequence generation, and candidate encoding.
"""

from typing import List, Optional, Tuple, Union
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerFast


class ELMDETokenizer:
    """
    Production Tokenizer wrapper for ELMDE.
    Uses Hugging Face AutoTokenizer (defaulting to answerdotai/ModernBERT-base or bert-base-uncased)
    configured for hardware-aligned sequence processing.
    """
    def __init__(
        self,
        model_name_or_path: str = "answerdotai/ModernBERT-base",
        fallback_model: str = "bert-base-uncased",
        max_length: int = 8192
    ):
        self.max_length = max_length
        self.model_name = model_name_or_path

        try:
            self.tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(model_name_or_path)
        except Exception:
            try:
                # Fallback to bert-base-uncased if offline or restricted
                self.tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(fallback_model)
                self.model_name = fallback_model
            except Exception:
                # Local standalone fallback using generic bert tokenizer
                from transformers import BertTokenizerFast
                self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
                self.model_name = "bert-base-uncased"

        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        self.cls_token_id = self.tokenizer.cls_token_id if self.tokenizer.cls_token_id is not None else 101
        self.sep_token_id = self.tokenizer.sep_token_id if self.tokenizer.sep_token_id is not None else 102
        self.vocab_size = len(self.tokenizer)

    def encode_text(
        self,
        text: Union[str, List[str]],
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = "pt"
    ) -> torch.Tensor:
        """Encodes single text or list of texts into input_ids tensor."""
        length = max_length or self.max_length
        encoded = self.tokenizer(
            text,
            max_length=length,
            truncation=True,
            padding=False,
            return_tensors=return_tensors
        )
        return encoded["input_ids"]

    def encode_jagged_stream(
        self,
        texts: List[str],
        max_length: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encodes a list of variable-length texts into a single contiguous jagged tensor
        with cumulative sequence length offsets (cu_seqlens) and token positions,
        completely eliminating padding tokens.

        Returns:
            input_ids_unpad: 1D Tensor of shape (N_total,)
            cu_seqlens: 1D Int Tensor of shape (B + 1,)
            positions: 1D Long Tensor of shape (N_total,)
        """
        tokenized_ids = [
            self.tokenizer.encode(t, max_length=max_length or self.max_length, truncation=True)
            for t in texts
        ]

        lengths = [len(ids) for ids in tokenized_ids]
        cu_seqlens = torch.zeros(len(texts) + 1, dtype=torch.int32)
        cu_seqlens[1:] = torch.cumsum(torch.tensor(lengths, dtype=torch.int32), dim=0)

        flat_ids = []
        positions_list = []
        for ids in tokenized_ids:
            flat_ids.extend(ids)
            positions_list.extend(range(len(ids)))

        input_ids_unpad = torch.tensor(flat_ids, dtype=torch.long)
        positions = torch.tensor(positions_list, dtype=torch.long)
        return input_ids_unpad, cu_seqlens, positions

    def decode_ids(self, token_ids: Union[List[int], torch.Tensor]) -> str:
        """Decodes token IDs back into string."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().tolist()
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
