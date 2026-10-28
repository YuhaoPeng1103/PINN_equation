"""nnschrodinger —— 用神经网络（PINN / Harrow-Hassidim-Lloyd 变体）求解与求值定态薛定谔方程。

顶层包只做元数据声明；具体实现位于各子模块，导入时不加载任何深度学习框架，
因此 `import nnschrodinger` 在没有 PyTorch 的环境里也能用于读取数据与参考解。
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
