# Remembrall

本页面介绍如何在LangChain中使用[Remembrall](https://remembrall.dev)生态系统。

## 什么是Remembrall？

Remembrall只需几行代码，就能为您的语言模型提供长期记忆、检索增强生成和完整的可观察性。

![Remembrall仪表板的截图，显示请求统计和模型交互。](/img/RemembrallDashboard.png "Remembrall仪表板界面")

它作为您的OpenAI调用之上的轻量级代理工作，只需在运行时用收集到的相关事实来增强聊天调用的上下文。

## 设置

要开始使用，请[在Remembrall平台上使用Github登录](https://remembrall.dev/login)并从[设置页面复制您的API密钥](https://remembrall.dev/dashboard/settings)。

任何您使用修改后的`openai_api_base`（见下文）和Remembrall API密钥发送的请求都会自动在Remembrall仪表板中被跟踪。您**永远不需要**与我们的平台共享您的OpenAI密钥，这些信息**永远不会**被Remembrall系统存储。

为此，我们需要安装以下依赖项：

```bash
pip install -U langchain-openai
```

### 启用长期记忆

除了通过`x-gp-api-key`设置`openai_api_base`和Remembrall API密钥外，您还应该指定一个UID来维护记忆。这通常是一个唯一的用户标识符（如电子邮件）。

```python
from langchain_openai import ChatOpenAI
chat_model = ChatOpenAI(openai_api_base="https://remembrall.dev/api/openai/v1",
                        model_kwargs={
                            "headers":{
                                "x-gp-api-key": "remembrall-api-key-here",
                                "x-gp-remember": "user@email.com",
                            }
                        })

chat_model.predict("My favorite color is blue.")
import time; time.sleep(5)  # 等待系统通过自动保存功能保存事实
print(chat_model.predict("What is my favorite color?"))
```

### 启用检索增强生成

首先，在[Remembrall仪表板](https://remembrall.dev/dashboard/spells)中创建一个文档上下文。粘贴文档文本或上传PDF文档进行处理。保存文档上下文ID并按如下所示插入。

```python
from langchain_openai import ChatOpenAI
chat_model = ChatOpenAI(openai_api_base="https://remembrall.dev/api/openai/v1",
                        model_kwargs={
                            "headers":{
                                "x-gp-api-key": "remembrall-api-key-here",
                                "x-gp-context": "document-context-id-goes-here",
                            }
                        })

print(chat_model.predict("This is a question that can be answered with my document."))
```
