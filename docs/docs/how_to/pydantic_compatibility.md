# 如何在不同的Pydantic版本下使用LangChain

从`0.3`版本开始，LangChain内部使用Pydantic 2。

用户应该安装Pydantic 2，并建议**避免**在LangChain API中使用Pydantic 2的`pydantic.v1`命名空间。

如果您正在使用早期版本的LangChain，请参阅以下关于[Pydantic兼容性](https://python.langchain.com/v0.2/docs/how_to/pydantic_compatibility)的指南。