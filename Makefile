# 便捷命令（Windows 上可用 make 兼容层，或直接复制命令执行）
PY ?= python

.PHONY: verify test lint fmt tree

verify:   ## 用独立数值解校验全部参考解（不需要 torch）
	$(PY) -m nnschrodinger.physics.verify

test:     ## 运行全部测试（无 torch 时网络相关测试自动跳过）
	$(PY) -m pytest

lint:
	ruff check src tests scripts

fmt:
	ruff format src tests scripts

tree:     ## 打印项目结构
	$(PY) scripts/tree.py
