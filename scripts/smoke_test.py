import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from wan2_1_unofficial import ModelConfig, UnofficialModel, reconstruction_loss


def main() -> None:
    config = ModelConfig(task="video", hidden_dim=64, num_layers=2, num_heads=4, vocab_size=2048)
    model = UnofficialModel(config)
    token_ids = torch.randint(0, config.vocab_size, (2, 12))
    image = torch.randn(2, 3, 32, 32)
    video = torch.randn(2, 3, 4, 32, 32)
    audio = torch.randn(2, 1, 1024)
    if config.task in {"video"}:
        out = model(token_ids=token_ids, video=video)
    elif config.task in {"audio"}:
        out = model(token_ids=token_ids, audio=audio)
    elif config.task in {"vlm"}:
        out = model(token_ids=token_ids, image=image)
    else:
        out = model(token_ids=token_ids)
    loss = reconstruction_loss(out["embedding"])
    print("sequence:", tuple(out["sequence"].shape))
    print("embedding:", tuple(out["embedding"].shape))
    print("loss:", float(loss))


if __name__ == "__main__":
    main()
