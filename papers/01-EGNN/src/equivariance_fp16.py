"""How much does fp16 break EGNN's rotation invariance?

Prints 3 numbers (max abs difference of the model output):
  fp32 equivariance error : fp32(pos)  vs fp32(rotated)   expect ~1e-6
  fp16 equivariance error : fp16(pos)  vs fp16(rotated)   expect ~1e-3
  fp16 rounding error     : fp32(pos)  vs fp16(pos)       scale for the line above

Run:  python papers/01-EGNN/src/equivariance_fp16.py [--run RUN_ID]
Without --run, uses a random-init model (invariance holds for any weights).
fp16 autocast needs CUDA; on CPU the fp16 lines just repeat fp32.
"""

import argparse
from pathlib import Path

import data
import torch
from model import EGNN

from gdl import load_checkpoint, load_run_config, run_dir

PAPER_DIR = Path(__file__).parent.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=str, default=None)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cuda = device == "cuda"

    _, _, test_loader, _, dims = data.load_split()

    if args.run:
        cfg = load_run_config(PAPER_DIR, args.run, base="runs")
        egnn = EGNN(
            in_node_nf=dims["in_node_dim"],
            hidden_nf=cfg["model"]["hidden_nf"],
            n_layers=cfg["model"]["n_layers"],
            attention=cfg["model"]["attention"],
        )
        load_checkpoint(egnn, run_dir(PAPER_DIR, args.run, base="runs"))
    else:
        torch.manual_seed(0)
        egnn = EGNN(
            in_node_nf=dims["in_node_dim"], hidden_nf=128, n_layers=7, attention=True
        )
    egnn = egnn.to(device).eval()

    b = next(iter(test_loader)).to(device)

    torch.manual_seed(1)
    Q, _ = torch.linalg.qr(torch.randn(3, 3))
    t = torch.randn(3)
    rotated = b.pos @ Q.T.to(device) + t.to(device)

    def predict(pos, fp16):
        with (
            torch.no_grad(),
            torch.autocast("cuda", dtype=torch.float16, enabled=fp16 and cuda),
        ):
            return egnn(b.x, pos, b.edge_index, batch=b.batch).float()

    a, a_rot = predict(b.pos, False), predict(rotated, False)
    h, h_rot = predict(b.pos, True), predict(rotated, True)

    def err(x, y):
        return (x - y).abs().max().item()

    print(f"device: {device}  (fp16 active: {cuda})")
    print(f"fp32 equivariance error : {err(a, a_rot):.2e}")
    print(f"fp16 equivariance error : {err(h, h_rot):.2e}")
    print(f"fp16 rounding error     : {err(a, h):.2e}")
    print(f"output scale (max |out|): {a.abs().max().item():.2e}")


if __name__ == "__main__":
    main()
