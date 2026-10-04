"""
配置加载工具
"""
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv


class ConfigLoader:
    """配置加载器"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        初始化配置加载器
        
        Args:
            config_path: 配置文件路径
        """
        # 加载环境变量
        load_dotenv()
        
        # 加载配置文件
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载YAML配置文件"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置项（支持点号分隔的嵌套键）
        
        Args:
            key: 配置键，如 "csrc.base_url"
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_env(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        获取环境变量
        
        Args:
            key: 环境变量名
            default: 默认值
            
        Returns:
            环境变量值
        """
        return os.getenv(key, default)
    
    def get_api_key(self, provider: str = "openai") -> Optional[str]:
        """
        获取API密钥
        
        Args:
            provider: 服务提供商
            
        Returns:
            API密钥
        """
        env_key = self.get(f"llm.api_key_env", f"{provider.upper()}_API_KEY")
        return self.get_env(env_key)
    
    @property
    def csrc_config(self) -> Dict[str, Any]:
        """获取证监会爬虫配置"""
        return self.config.get("csrc", {})
    
    @property
    def astock_config(self) -> Dict[str, Any]:
        """获取A股数据配置"""
        return self.config.get("astock", {})
    
    @property
    def crypto_config(self) -> Dict[str, Any]:
        """获取加密货币配置"""
        return self.config.get("crypto", {})
    
    @property
    def labeling_config(self) -> Dict[str, Any]:
        """获取标注规则配置"""
        return self.config.get("labeling", {})
    
    @property
    def paths_config(self) -> Dict[str, Any]:
        """获取路径配置"""
        return self.config.get("paths", {})


# 全局配置实例
config = ConfigLoader()

