"""测试全局配置：在导入应用前关闭业务 JSONL 落盘、隔离日志目录，避免污染真实 logs/。"""

import os

os.environ["BUSINESS_LOG_ENABLED"] = "0"
os.environ.setdefault("LOG_DIR", "/tmp/yolo-service-test-logs")
