# Streamlit

> **[Streamlit](https://streamlit.io/) 是一种更快地构建和分享数据应用的方式。**
> Streamlit 可以在几分钟内将数据脚本转变为可共享的网络应用。完全使用纯 Python。无需前端经验。
> 在 [streamlit.io/generative-ai](https://streamlit.io/generative-ai) 查看更多示例。

[![在 GitHub Codespaces 中打开](https://github.com/codespaces/badge.svg)](https://codespaces.new/langchain-ai/streamlit-agent?quickstart=1)

在本指南中，我们将演示如何使用 `StreamlitCallbackHandler` 在交互式 Streamlit 应用中显示代理的思考和行动。通过下方运行的应用使用 MRKL 代理来体验它：

<iframe loading="lazy" src="https://langchain-mrkl.streamlit.app/?embed=true&embed_options=light_theme"
    style={{ width: 100 + '%', border: 'none', marginBottom: 1 + 'rem', height: 600 }}
    allow="camera;clipboard-read;clipboard-write;"
></iframe>

## 安装和设置

```bash
pip install langchain streamlit
```

您可以运行 `streamlit hello` 来加载示例应用并验证您的安装是否成功。完整说明请参见 Streamlit 的[入门文档](https://docs.streamlit.io/library/get-started)。

## 显示思考和行动

要创建 `StreamlitCallbackHandler`，您只需提供一个用于渲染输出的父容器。

```python
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)
import streamlit as st

st_callback = StreamlitCallbackHandler(st.container())
```

自定义显示行为的其他关键字参数在[API 参考](https://python.langchain.com/api_reference/langchain/callbacks/langchain.callbacks.streamlit.streamlit_callback_handler.StreamlitCallbackHandler.html)中有描述。

### 场景 1：使用带工具的代理

目前主要支持的用例是可视化带工具代理（或代理执行器）的行动。您可以在 Streamlit 应用中创建一个代理，只需将 `StreamlitCallbackHandler` 传递给 `agent.run()` 即可在应用中实时可视化思考和行动。

```python
import streamlit as st
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, load_tools
from langchain_openai import OpenAI

llm = OpenAI(temperature=0, streaming=True)
tools = load_tools(["ddg-search"])
prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent_executor.invoke(
            {"input": prompt}, {"callbacks": [st_callback]}
        )
        st.write(response["output"])
```

**注意：** 您需要设置 `OPENAI_API_KEY` 才能成功运行上述应用代码。
最简单的方法是通过 [Streamlit secrets.toml](https://docs.streamlit.io/library/advanced-features/secrets-management)，
或任何其他本地环境变量管理工具。

### 其他场景

目前 `StreamlitCallbackHandler` 主要针对与 LangChain 代理执行器一起使用。未来将添加对额外代理类型、直接与链一起使用等的支持。

您可能还对使用 [StreamlitChatMessageHistory](/docs/integrations/memory/streamlit_chat_message_history) 与 LangChain 一起使用感兴趣。
