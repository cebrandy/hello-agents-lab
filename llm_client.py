"""Hello-Agents 教程中的基础 LLM 客户端。"""

import os
from typing import Dict, List, Optional
from urllib.parse import urlparse
from uuid import uuid4

from dotenv import load_dotenv
from openai import OpenAI


# 从当前项目目录的 .env 文件加载模型配置。
load_dotenv()


class HelloAgentsLLM:
    """调用任何兼容 OpenAI API 的模型服务，默认使用流式响应。"""

    def __init__(
        self,
        model: Optional[str] = None,
        apiKey: Optional[str] = None,
        baseUrl: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        """优先使用传入参数；未提供时从环境变量读取。"""
        self.model = model or os.getenv("LLM_MODEL_ID")
        api_key = apiKey or os.getenv("LLM_API_KEY")
        base_url = baseUrl or os.getenv("LLM_BASE_URL")
        request_timeout = timeout or int(os.getenv("LLM_TIMEOUT", "60"))

        if not all((self.model, api_key, base_url)):
            raise ValueError(
                "模型 ID、API 密钥和服务地址必须提供，或在项目根目录的 .env 文件中配置。"
            )

        default_headers = None
        parsed_base_url = urlparse(base_url)
        if (
            parsed_base_url.hostname == "opencode.ai"
            and parsed_base_url.path.rstrip("/") == "/zen/go/v1"
        ):
            # OpenCode Go 要求同一次对话的请求共用一个稳定的会话 ID。
            default_headers = {
                "User-Agent": "hello-agents-tutorial/1.0",
                "x-opencode-session": str(uuid4()),
                # 当前 Python 环境的 HTTP 解压库与压缩响应不兼容。
                "Accept-Encoding": "identity",
            }

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=request_timeout,
            default_headers=default_headers,
        )

    def think(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0,
    ) -> str:
        """发送对话消息，边接收边打印模型回复，并返回完整文本。"""
        print(f"🧠 正在调用 {self.model} 模型...")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            stream=True,
        )

        collected_content: List[str] = []
        for chunk in response:
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content or ""
            print(content, end="", flush=True)
            collected_content.append(content)

        print()
        return "".join(collected_content)
