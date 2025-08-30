from nekro_agent.services.plugin.base import ConfigBase, NekroPlugin
from pydantic import Field


# 创建插件实例
plugin = NekroPlugin(
    name="stable_diffusion绘图",
    module_name="stable_diffusion_draw",
    description="基于stable-diffusion api的绘图插件,支持文生图和图生图。",
    version="0.1.1", 
    author="ZerBC",
    url="https://github.com/ZerBC/nekro_plugin_SDdraw",
)


@plugin.mount_config()
class StableDiffusionDrawConfig(ConfigBase):
    """Stable Diffusion 绘图插件配置

    定义了Stable Diffusion插件运行所需的各项配置，包括API地址、超时设置以及最大图像尺寸。
    """

    API_URL: str = Field(
        default="http://127.0.0.1:7860",
        title="API地址",
        description="Stable Diffusion API地址，默认127.0.0.1:7860，请确保包含协议头（如http://）。",
    )
    TIMEOUT: int = Field(
        default=300,
        title="绘图超时时间",
        description="单位: 秒，用于API请求的超时设置。",
        ge=1,  # 超时时间必须大于等于1秒
    )
    MAX_IMAGE_DIMENSION: int = Field(
        default=1080,
        title="图像最大维度",
        description="图像宽度或高度的最大允许像素值。例如，如果设置为2048，则宽度和高度都不能超过2048像素。",
        ge=256,  # 维度必须大于等于256
    )


# 获取配置实例
config = plugin.get_config(StableDiffusionDrawConfig)
# 获取插件存储
store = plugin.store