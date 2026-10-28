"""求解器层：采样、网络、损失、训练与评测。

依赖 PyTorch 的模块（networks / losses / train）在函数内部延迟导入，
因此 `import nnschrodinger.solver` 本身不需要 torch。
"""

__all__ = ["inputs", "networks", "losses", "train", "evaluate", "backends"]
