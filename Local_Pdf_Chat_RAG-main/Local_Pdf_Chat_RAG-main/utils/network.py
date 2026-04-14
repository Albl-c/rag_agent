"""
网络工具 —— HTTP Session 管理、端口检测
"""

import socket
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_session = None


def get_session():
    """获取带重试机制的 HTTP Session（单例）"""
    global _session
    if _session is None:
        _session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=0.1,
            status_forcelist=[500, 502, 503, 504]
        )
        _session.mount('http://', HTTPAdapter(max_retries=retries))
    return _session


def is_port_available(port):
    """检测端口是否可用"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(('127.0.0.1', port)) != 0
