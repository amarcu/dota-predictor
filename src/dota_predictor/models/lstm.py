"""
LSTM-based model for Dota 2 match prediction.

Based on the architecture from arXiv:2106.01782, this model uses LSTM
layers to process the time-series game state data and predict match outcomes.
"""

import torch
import torch.nn as nn

from dota_predictor.utils.config import (
    DROPOUT,
    HERO_EMBEDDING_DIM,
    HIDDEN_SIZE,
    INPUT_SIZE,
    NUM_HEROES,
    NUM_LAYERS,
)


class LSTMPredictor(nn.Module):
    """
    LSTM-based match outcome predictor.

    Architecture:
    1. Optional hero embedding layer
    2. LSTM layers for time-series processing
    3. Fully connected layers for classification

    The model can predict:
    - Final match outcome (binary classification)
    - Win probability at each timestep (sequence output)
    """

    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
        num_heroes: int = NUM_HEROES,
        hero_embedding_dim: int = HERO_EMBEDDING_DIM,
        use_hero_embedding: bool = True,
        bidirectional: bool = False,
    ) -> None:
        """
        Initialize the LSTM predictor.

        Args:
            input_size: Number of input features per timestep
            hidden_size: LSTM hidden state size
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            num_heroes: Number of heroes for embedding
            hero_embedding_dim: Hero embedding dimension
            use_hero_embedding: Whether to use hero embeddings
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_hero_embedding = use_hero_embedding
        self.bidirectional = bidirectional

        # Hero embedding
        if use_hero_embedding:
            self.hero_embedding = nn.Embedding(
                num_embeddings=num_heroes + 1,  # +1 for padding/unknown
                embedding_dim=hero_embedding_dim,
                padding_idx=0,
            )
            # 10 heroes * embedding_dim
            hero_feature_size = 10 * hero_embedding_dim
        else:
            self.hero_embedding = None
            hero_feature_size = 0

        # LSTM for time-series
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Direction multiplier for hidden size
        direction_mult = 2 if bidirectional else 1

        # Fully connected layers
        fc_input_size = hidden_size * direction_mult + hero_feature_size
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        # For sequence output (predict at each timestep) — includes hero features
        self.seq_fc = nn.Linear(hidden_size * direction_mult + hero_feature_size, 1)

    def forward(
        self,
        features: torch.Tensor,
        heroes: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        return_sequence: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features: Time-series features (batch, seq_len, input_size)
            heroes: Hero IDs (batch, 10) - 5 radiant + 5 dire
            mask: Valid timestep mask (batch, seq_len)
            return_sequence: If True, return predictions for each timestep

        Returns:
            If return_sequence=False: (batch,) win probabilities
            If return_sequence=True: (batch, seq_len) win probabilities per timestep
        """
        batch_size = features.size(0)

        # Process time-series with LSTM
        lstm_out, (hidden, _) = self.lstm(features)
        # lstm_out: (batch, seq_len, hidden_size * directions)
        # hidden: (num_layers * directions, batch, hidden_size)

        if return_sequence:
            # Predict at each timestep, including hero context broadcast across all steps
            if self.use_hero_embedding and heroes is not None:
                hero_embeds = self.hero_embedding(heroes)          # (batch, 10, embed_dim)
                hero_features = hero_embeds.view(batch_size, -1)  # (batch, 320)
                seq_len = lstm_out.size(1)
                hero_features_exp = hero_features.unsqueeze(1).expand(-1, seq_len, -1)
                lstm_combined = torch.cat([lstm_out, hero_features_exp], dim=-1)
            else:
                lstm_combined = lstm_out
            seq_logits = self.seq_fc(lstm_combined).squeeze(-1)  # (batch, seq_len)
            return torch.sigmoid(seq_logits)

        # Get final hidden state
        if self.bidirectional:
            # Concatenate forward and backward final hidden states
            final_hidden = torch.cat(
                [hidden[-2], hidden[-1]],
                dim=-1,
            )  # (batch, hidden_size * 2)
        else:
            final_hidden = hidden[-1]  # (batch, hidden_size)

        # Add hero features
        if self.use_hero_embedding and heroes is not None:
            hero_embeds = self.hero_embedding(heroes)  # (batch, 10, embed_dim)
            hero_features = hero_embeds.view(batch_size, -1)  # (batch, 10 * embed_dim)
            combined = torch.cat([final_hidden, hero_features], dim=-1)
        else:
            combined = final_hidden

        # Final prediction
        logits = self.fc(combined).squeeze(-1)  # (batch,)
        return torch.sigmoid(logits)

    def predict_at_time(
        self,
        features: torch.Tensor,
        minute: int,
        heroes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Predict win probability at a specific minute.

        This is useful for multi-forward step prediction as described
        in the paper - we can predict the outcome at various game stages.

        Args:
            features: Full time-series features
            minute: The minute to predict at
            heroes: Optional hero IDs

        Returns:
            Win probability at the specified minute
        """
        masked_features = features.clone()
        masked_features[:, minute:, :] = 0

        seq_probs = self.forward(masked_features, heroes, return_sequence=True)
        return seq_probs[:, min(minute - 1, seq_probs.size(1) - 1)]


class LSTMWithAttention(nn.Module):
    """
    LSTM with attention mechanism for better interpretability.

    This variant adds an attention layer that learns to focus on
    important moments in the game (e.g., team fight outcomes,
    objective takes, etc.)
    """

    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
    ) -> None:
        """Initialize LSTM with attention."""
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with attention.

        Args:
            features: Time-series features (batch, seq_len, input_size)
            mask: Valid timestep mask (batch, seq_len)

        Returns:
            Tuple of (predictions, attention_weights)
        """
        # LSTM encoding
        lstm_out, _ = self.lstm(features)  # (batch, seq_len, hidden)

        # Attention weights
        attn_scores = self.attention(lstm_out).squeeze(-1)  # (batch, seq_len)

        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)  # (batch, seq_len)

        # Weighted sum
        context = torch.bmm(
            attn_weights.unsqueeze(1),
            lstm_out,
        ).squeeze(1)  # (batch, hidden)

        # Prediction
        logits = self.fc(context).squeeze(-1)
        probs = torch.sigmoid(logits)

        return probs, attn_weights

