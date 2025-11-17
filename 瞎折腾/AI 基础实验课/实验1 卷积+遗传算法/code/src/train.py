from pathlib import Path
from typing import Optional, Union
import json
import torch
import torch.nn as nn
import config as C



def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_simple_and_save(
        model: torch.nn.Module,
        train_loader,
        test_loader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
    device: Optional[torch.device] = None,
    save_dir: Union[str, Path] = C.OUTPUT_DIR,
        id: str = "temp",
    params: Optional[dict] = None):
    """
    简单训练循环，返回 (最终train_loss, 最终test_loss)。
    保存日志到 save_dir/id/train_log.json。
    """
    device = device or get_device()
    save_base = Path(save_dir)
    save_path = save_base / str(id)
    save_path.mkdir(parents=True, exist_ok=True)

    log = {
        "params": params or {},
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "model_structure": str(model),
    }

    model.to(device)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()

            bs = y.size(0)
            total_loss += loss.item() * bs
            preds = outputs.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += bs

        train_loss = total_loss / max(1, total_samples)
        train_acc = total_correct / max(1, total_samples)
        log["train_loss"].append(float(train_loss))
        log["train_acc"].append(float(train_acc))

        # eval
        model.eval()
        t_loss = 0.0
        t_correct = 0
        t_samples = 0
        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(device)
                y = y.to(device)
                out = model(X)
                l = loss_fn(out, y)
                bs = y.size(0)
                t_loss += l.item() * bs
                t_correct += (out.argmax(dim=1) == y).sum().item()
                t_samples += bs

        test_loss = t_loss / max(1, t_samples)
        test_acc = t_correct / max(1, t_samples)
        log["test_loss"].append(float(test_loss))
        log["test_acc"].append(float(test_acc))

    print(
        f"train_loss={train_loss:.4f}  test_loss={test_loss:.4f} | "
        f"train_acc={train_acc:.4f}  test_acc={test_acc:.4f}\n"
    )

    # 保存日志
    json_log = {
        "params": log["params"],
        "train_loss": [float(x) for x in log["train_loss"]],
        "train_acc": [float(x) for x in log["train_acc"]],
        "test_loss": [float(x) for x in log["test_loss"]],
        "test_acc": [float(x) for x in log["test_acc"]],
        "model_structure": log["model_structure"],
        "final_test_loss": float(log["test_loss"][-1]) if log["test_loss"] else None,
        "final_test_acc": float(log["test_acc"][-1]) if log["test_acc"] else None,
    }
    with open(save_path / "train_log.json", "w", encoding="utf-8") as f:
        json.dump(json_log, f, indent=2, ensure_ascii=False)

    return float(train_loss), float(json_log["final_test_loss"])