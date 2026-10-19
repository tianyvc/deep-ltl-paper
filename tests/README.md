# 检查脚本

从仓库根目录设置 PYTHONPATH=src 后运行 test_gnn_embedding_quick.py 和 test_gnn_fix.py。前者是直接执行的 smoke script，不是标准 pytest 测试函数集合；后者检查 GNN 初始化和前向。

legacy/ 保存历史脚本以便追溯，存在旧 API、缺失 checkpoint、占位评估以及捕获异常继续等问题，不能作为已通过测试。最新实际结果见 ../docs/maintenance/VALIDATION.md。
