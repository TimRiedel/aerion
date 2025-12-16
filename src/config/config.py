from typing import Any, Dict, Optional
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.global_hydra import GlobalHydra


class Config:
    """Configuration manager using Hydra and OmegaConf."""
    
    @staticmethod
    def load_config(config_path: Optional[str] = None) -> DictConfig:
        """Load configuration from Hydra."""
        if GlobalHydra.instance().is_initialized():
            cfg = hydra.core.hydra_config.HydraConfig.get()
            return cfg
        raise RuntimeError("Hydra not initialized. Use @hydra.main decorator.")
    
    @staticmethod
    def get_config_value(cfg: DictConfig, key: str, default: Any = None) -> Any:
        """Get a configuration value with optional default."""
        return OmegaConf.select(cfg, key, default=default)
    
    @staticmethod
    def merge_configs(base: DictConfig, override: DictConfig) -> DictConfig:
        """Merge two configurations."""
        return OmegaConf.merge(base, override)
    
    @staticmethod
    def to_dict(cfg: DictConfig) -> Dict[str, Any]:
        """Convert OmegaConf DictConfig to regular dict."""
        return OmegaConf.to_container(cfg, resolve=True)

