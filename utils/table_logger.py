"""
Shared Excel table logger with append-or-overwrite-by-model-name logic.

设计原则：
  - 每个"实验"对应一个 xlsx 文件
  - 第一列固定为 "Model"（模型名），作为主键
  - update() 时：若 Model 已存在 → 覆盖该行；否则 → 追加新行
  - 每个 TableLogger 实例绑定一个 xlsx 文件

Usage:
    log = TableLogger("outputs/freihand/tables/main_comparison.xlsx")
    log.update("MD-HCRNet", {"MPJPE (mm)": 12.3, "PA-MPJPE (mm)": 8.1})
    log.update("I2L-Baseline", {"MPJPE (mm)": 18.4, "PA-MPJPE (mm)": 12.5})
"""
import os
import pandas as pd


class TableLogger:
    """
    Multi-model Excel table logger.

    Args:
        path:    absolute path to the .xlsx file
        key_col: name of the primary-key column (default "Model")
    """

    def __init__(self, path: str, key_col: str = "Model"):
        self.path    = path
        self.key_col = key_col
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    # ── internal helpers ──────────────────────────────────────────────────────

    def _load(self) -> pd.DataFrame:
        if os.path.exists(self.path):
            try:
                return pd.read_excel(self.path, engine="openpyxl")
            except Exception:
                pass
        return pd.DataFrame(columns=[self.key_col])

    def _save(self, df: pd.DataFrame):
        df.to_excel(self.path, index=False, engine="openpyxl")

    # ── public API ────────────────────────────────────────────────────────────

    def update(self, model_name: str, data: dict):
        """
        Add or update the row whose first column equals model_name.

        Args:
            model_name: identifier (e.g. "MD-HCRNet", "I2L-Baseline")
            data:       dict of {column_name: value}
        """
        row = {self.key_col: model_name, **data}
        df  = self._load()

        # ensure all new columns exist
        for col in row:
            if col not in df.columns:
                df[col] = None

        mask = df[self.key_col].astype(str) == str(model_name)
        if mask.any():
            for k, v in row.items():
                df.loc[mask, k] = v
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

        self._save(df)

    def read(self) -> pd.DataFrame:
        """Return the current table as a DataFrame."""
        return self._load()

    def model_names(self) -> list:
        df = self._load()
        return df[self.key_col].tolist() if self.key_col in df.columns else []


# ── convenience: multi-epoch training logger ─────────────────────────────────

class TrainingCurveLogger:
    """
    Logs per-epoch training/val loss for multiple models into one xlsx.

    Columns: Model | Epoch | TrainLoss | ValLoss | LR | BatchMS
    Key:     (Model, Epoch) pair — overwrites if already present.
    """

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def _load(self) -> pd.DataFrame:
        if os.path.exists(self.path):
            try:
                return pd.read_excel(self.path, engine="openpyxl")
            except Exception:
                pass
        return pd.DataFrame(columns=["Model", "Epoch", "TrainLoss", "ValLoss", "LR", "BatchMS"])

    def log(self, model_name: str, epoch: int, train_loss: float,
            val_loss: float, lr: float = None, batch_ms: float = None):
        """Append or overwrite (model_name, epoch) row."""
        row = {
            "Model":      model_name,
            "Epoch":      epoch + 1,
            "TrainLoss":  round(train_loss, 6),
            "ValLoss":    round(val_loss,   6),
            "LR":         round(lr, 8) if lr is not None else None,
            "BatchMS":    round(batch_ms, 1) if batch_ms is not None else None,
        }
        df   = self._load()
        mask = (df["Model"].astype(str) == str(model_name)) & (df["Epoch"] == epoch + 1)
        if mask.any():
            for k, v in row.items():
                df.loc[mask, k] = v
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_excel(self.path, index=False, engine="openpyxl")

    def clear_model(self, model_name: str):
        """Remove all curve rows for one model before a fresh training run."""
        df = self._load()
        if "Model" not in df.columns:
            return
        df = df[df["Model"].astype(str) != str(model_name)]
        df.to_excel(self.path, index=False, engine="openpyxl")
