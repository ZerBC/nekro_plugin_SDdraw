import base64
from typing import Any, Dict, Union

import httpx
from pydantic import BaseModel, Field, ValidationError

from nekro_agent.api.schemas import AgentCtx
from nekro_agent.core import logger
from nekro_agent.services.plugin.base import (
    SandboxMethodType,
)

from .conf import config, plugin

# 内部 Pydantic 模型，用于验证和解析 generate_image 方法的输入参数
class ImageGenerationRequestParams(BaseModel):
    """
    用于验证和解析图像生成请求参数的内部模型。
    确保所有必要的参数都存在且格式正确。
    """
    p_prompt: str = Field(description="正向提示词：描述所有您希望出现在图像中的元素，尽可能详细，使用英文。")
    n_prompt: str = Field(description="负向提示词：描述所有您不希望出现在图像中的元素，尽可能详细，使用英文。")
    size: str = Field(description="图像尺寸，格式为 '宽*高'，例如 '640*980'。")




@plugin.mount_sandbox_method(
    SandboxMethodType.TOOL,  
    name="生成图片",
    description="使用 Stable Diffusion API 生成图片，并直接返回图片数据。", 
)
async def generate_image(
    _ctx: AgentCtx,
    p_prompt: str,
    n_prompt: str,
    size: str,
) -> bytes: # 修改: 返回值类型改为 bytes
    """使用 Stable Diffusion API 生成图片，并直接返回PNG格式图片二进制数据。
    根据对话者说的话推断他想要的图像，生成下面的提示词。
    函数执行完毕后需将返回的二进制数据转成图片发出。

    Args:
        p_prompt: 正向提示词 (Positive Prompt)，详细描述希望出现在图像中的元素、风格、场景等，
                  使用英文表达，越详细越好。
        n_prompt: 负向提示词 (Negative Prompt)，详细描述所有不希望出现在图像中的元素、缺陷等，
                  使用英文表达，越详细越好。
        size: 图像尺寸，格式为 '宽*高'。必须是整数。

    Returns:
        bytes: 生成的PNG格式图片二进制数据。

    Example:
        generate_image(
            p_prompt="1girl, grey hair, red eyes, grey uniform, floating hair, masterpiece, best quality",
            n_prompt="blurry, ugly, distorted, low quality, bad anatomy, deformed, worst quality",
            size="640*960"
        )
        with open('./shared/anime_girl.png', 'wb') as f:
            f.write(image_data)
        send_msg_file(_ck, './shared/anime_girl.png')
    """
    # 1. 输入参数验证和解析
    try:
        params = ImageGenerationRequestParams(p_prompt=p_prompt, n_prompt=n_prompt, size=size)
        width_str, height_str = params.size.split('*')
        width, height = int(width_str), int(height_str)
        if not (0 < width <= config.MAX_IMAGE_DIMENSION and 0 < height <= config.MAX_IMAGE_DIMENSION):
            raise ValueError(
                f"请求的图像尺寸 {width}x{height} 无效或超出最大限制 "
                f"({config.MAX_IMAGE_DIMENSION}x{config.MAX_IMAGE_DIMENSION})。"
            )
    except ValueError as e:
        logger.warning(f"绘图尺寸参数无效: size='{size}'. 错误: {e}")
        # 修改: 抛出异常而不是返回字符串，由框架统一处理错误
        raise ValueError(f"绘图失败：尺寸格式 '{size}' 不正确或超出范围。请使用 '宽*高' 格式，并确保不超出最大限制。")
    except ValidationError as e:
        error_messages = "; ".join([f"{err['loc'][0]}: {err['msg']}" for err in e.errors()])
        logger.warning(f"绘图参数验证失败: {error_messages}")
        raise ValueError(f"绘图参数错误：{error_messages}")

    # 2. 构建 Stable Diffusion API 请求体
    api_endpoint = f"{config.API_URL.rstrip('/')}/sdapi/v1/txt2img"
    payload: Dict[str, Any] = {
        "prompt": params.p_prompt,
        "negative_prompt": params.n_prompt,
        "width": width,
        "height": height,
        "steps": 20, "sampler_name": "DPM++ 2M Karras", "cfg_scale": 7,
        "batch_size": 1, "n_iter": 1, "seed": -1,
        "send_images": True, "save_images": False,
    }

    # 3. 发送 API 请求并处理响应
    try:
        async with httpx.AsyncClient(timeout=config.TIMEOUT) as client:
            response = await client.post(api_endpoint, json=payload)
            response.raise_for_status()
            data = response.json()

            if not data.get("images") or not data["images"][0]:
                logger.warning("Stable Diffusion API响应中未找到图片数据。响应: %s", data)
                raise Exception("绘图失败：API响应中未包含有效的图片数据。")

            image_base64: str = data["images"][0]
            image_data: bytes = base64.b64decode(image_base64)
            
            logger.info("图片已成功生成，将返回图片二进制数据。")
            return image_data

    except httpx.RequestError as e:
        logger.error(f"无法连接到Stable Diffusion服务: {e}", exc_info=True)
        raise Exception("绘图失败：无法连接到Stable Diffusion服务。请检查API地址和网络设置。")
    except httpx.HTTPStatusError as e:
        logger.error(f"Stable Diffusion服务返回错误: {e.response.status_code} - {e.response.text}", exc_info=True)
        raise Exception(f"绘图失败：服务返回错误 {e.response.status_code}。请检查API服务日志。")
    except (KeyError, IndexError, ValueError) as e:
        logger.error(f"API响应数据解析错误: {e}", exc_info=True)
        raise Exception("绘图失败：无法解析来自API的响应数据。")
    except base64.binascii.Error as e:
        logger.error(f"绘图失败：无法解码Base64图片数据。错误: {e}", exc_info=True)
        raise Exception("绘图失败：收到的图片数据格式不正确，无法解码。")
    except Exception as e:
        # 捕获所有其他潜在的异常
        logger.error(f"绘图时发生未知错误: {e}", exc_info=True)
        raise Exception("绘图时发生未知错误，请联系管理员。")


@plugin.mount_cleanup_method()
async def clean_up():
    """清理插件资源。此插件无需特殊清理操作。"""
    logger.info("Stable Diffusion 绘图插件资源已清理。")
