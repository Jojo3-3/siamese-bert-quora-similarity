"""Siamese BERT model for semantic duplicate detection."""

import torch
from torch import nn
from transformers import AutoModel


class SiameseBERT(nn.Module):
    """Encode two questions with shared weights and classify their L1 distance."""

    def __init__(self, model_name: str, dropout_rate: float = 0.2) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
        )

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return a mean-pooled sentence embedding while ignoring padding."""
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        mask = attention_mask.unsqueeze(-1).to(outputs.last_hidden_state.dtype)
        summed = (outputs.last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def forward(
        self,
        q1_input_ids: torch.Tensor,
        q1_attention_mask: torch.Tensor,
        q2_input_ids: torch.Tensor,
        q2_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        q1_embedding = self.encode(q1_input_ids, q1_attention_mask)
        q2_embedding = self.encode(q2_input_ids, q2_attention_mask)
        l1_distance = torch.abs(q1_embedding - q2_embedding)
        return self.classifier(l1_distance).squeeze(-1)
