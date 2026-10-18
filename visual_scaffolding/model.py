"""Visual scaffolding modules and strict checkpoint loading for inference."""

import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration

from .retrieval import RetrievalModule

SPECIAL_TOKENS = {
    "request": "<|img_ref|>",
    "evidence": "<|ref_pad|>",
    "end": "<|ref_end|>",
}


class EvidenceAdaptor(nn.Module):
    """Preserve native features while adapting them to the reasoning sequence."""

    def __init__(self, width=2560):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(width, width)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, features):
        features = features.to(dtype=self.fc1.weight.dtype)
        return features + self.fc2(self.act(self.fc1(self.norm(features))))


class VisualScaffoldModel(Qwen3VLForConditionalGeneration):
    """Qwen3-VL with the visual retriever and residual evidence adaptor."""

    def __init__(self, config):
        if config.text_config.hidden_size != 2560:
            raise ValueError("This release expects the supplied 4B checkpoints")
        super().__init__(config)
        self.retrieval_module = RetrievalModule(
            query_hidden_size=2560,
            key_hidden_size=2560,
            score_dim=2048,
            spatial_dim=512,
            num_heads=16,
            context_num_heads=8,
        )
        self.evidence_mlp = EvidenceAdaptor()


def load_checkpoint(checkpoint, device="cuda:0"):
    """Load all model weights and validate the tokenizer's evidence tokens."""
    checkpoint = Path(checkpoint).expanduser().resolve()
    settings = json.loads((checkpoint / "scaffold_config.json").read_text())
    expected = {
        "hidden_size": 2560,
        "score_dim": 2048,
        "spatial_dim": 512,
        "retrieval_heads": 16,
        "context_heads": 8,
        "context_window": 256,
        "image_scale": 2,
    }
    if settings != expected:
        raise ValueError("Checkpoint architecture does not match this release")
    model, info = VisualScaffoldModel.from_pretrained(
        str(checkpoint),
        dtype=torch.bfloat16,
        device_map=None,
        output_loading_info=True,
    )
    if any(
        info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise ValueError(f"Checkpoint did not load completely: {info}")
    model.to(device=device, dtype=torch.bfloat16).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), trust_remote_code=False)
    processor = AutoProcessor.from_pretrained(str(checkpoint), trust_remote_code=False)
    token_ids = {
        name: tokenizer.convert_tokens_to_ids(token) for name, token in SPECIAL_TOKENS.items()
    }
    vocab = model.get_input_embeddings().weight.shape[0]
    if any(value is None or value < 0 or value >= vocab for value in token_ids.values()):
        raise ValueError("Checkpoint tokenizer is missing the trained evidence tokens")
    if len(set(token_ids.values())) != 3:
        raise ValueError("Evidence tokens must have distinct IDs")
    for name, token in SPECIAL_TOKENS.items():
        if processor.tokenizer.convert_tokens_to_ids(token) != token_ids[name]:
            raise ValueError(f"Processor and model disagree about token {token}")
    return model, processor, tokenizer, token_ids
