"""
Production Dataset & DataCollator for ELMDE.
Supports unpadded jagged batching, multi-candidate discrete choices,
CORN ordinal ranking, and continuous density targets.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import torch
from torch.utils.data import Dataset

from .tokenizer import ELMDETokenizer


@dataclass
class DecisionSample:
    """Represents a single decision instance."""
    context_text: str
    target_choice: Optional[int] = None
    target_ordinal: Optional[int] = None
    target_density: Optional[float] = None
    candidate_texts: Optional[List[str]] = None
    valid_candidate_mask: Optional[List[bool]] = None
    metadata: Optional[Dict[str, Any]] = None


class DecisionDataset(Dataset):
    """
    Production PyTorch Dataset for ELMDE.
    Loads from in-memory samples, list of dictionaries, or JSONL files.
    """
    def __init__(self, samples: List[Union[DecisionSample, Dict[str, Any]]]):
        self.samples: List[DecisionSample] = []
        for s in samples:
            if isinstance(s, DecisionSample):
                self.samples.append(s)
            elif isinstance(s, dict):
                self.samples.append(DecisionSample(
                    context_text=s["context_text"],
                    target_choice=s.get("target_choice"),
                    target_ordinal=s.get("target_ordinal"),
                    target_density=s.get("target_density"),
                    candidate_texts=s.get("candidate_texts"),
                    valid_candidate_mask=s.get("valid_candidate_mask"),
                    metadata=s.get("metadata")
                ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> DecisionSample:
        return self.samples[idx]

    @classmethod
    def from_jsonl(cls, filepath: str) -> "DecisionDataset":
        """Loads samples from a JSONL file."""
        samples = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return cls(samples)


class JaggedDataCollator:
    """
    Collate function for PyTorch DataLoader that packs a minibatch of variable-length
    sequences into unpadded jagged streams with cu_seqlens and positions,
    eliminating compute waste on padding tokens.
    """
    def __init__(self, tokenizer: ELMDETokenizer, max_length: int = 4096):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[DecisionSample]) -> Dict[str, Any]:
        texts = [s.context_text for s in batch]
        input_ids_unpad, cu_seqlens, positions = self.tokenizer.encode_jagged_stream(
            texts, max_length=self.max_length
        )

        # Standard padded batch fallback for modules requiring rectangular shapes
        # (pads to max length in current batch)
        encoded_padded = self.tokenizer.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        batch_dict: Dict[str, Any] = {
            "input_ids": encoded_padded["input_ids"],
            "attention_mask": encoded_padded["attention_mask"],
            "input_ids_unpad": input_ids_unpad,
            "cu_seqlens": cu_seqlens,
            "positions": positions,
            "batch_size": len(batch)
        }

        # Collect targets
        if batch[0].target_choice is not None:
            batch_dict["target_choice"] = torch.tensor(
                [s.target_choice for s in batch], dtype=torch.long
            )

        if batch[0].target_ordinal is not None:
            batch_dict["target_ordinal"] = torch.tensor(
                [s.target_ordinal for s in batch], dtype=torch.long
            )

        if batch[0].target_density is not None:
            batch_dict["target_density"] = torch.tensor(
                [s.target_density for s in batch], dtype=torch.float32
            )

        # Candidate texts if present
        if batch[0].candidate_texts is not None:
            batch_dict["candidate_texts"] = [s.candidate_texts for s in batch]
            if batch[0].valid_candidate_mask is not None:
                batch_dict["valid_mask"] = torch.tensor(
                    [s.valid_candidate_mask for s in batch], dtype=torch.bool
                )

        return batch_dict
