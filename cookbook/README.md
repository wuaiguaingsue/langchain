# LangChain cookbook（LangChain 操作指南）

使用 LangChain 构建应用程序的示例代码，相比[主文档](https://python.langchain.com)更侧重于实际应用和端到端示例。

笔记本 | 描述
:- | :-
[agent_fireworks_ai_langchain_mongodb.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/agent_fireworks_ai_langchain_mongodb.ipynb) | 使用 MongoDB、LangChain 和 FireWorksAI 构建具有记忆功能的 AI 代理。
[mongodb-langchain-cache-memory.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/mongodb-langchain-cache-memory.ipynb) | 使用 MongoDB 和 LangChain 构建具有语义缓存的 RAG 应用程序。
[LLaMA2_sql_chat.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/LLaMA2_sql_chat.ipynb) | 构建一个使用开源语言模型(llama2)与 SQL 数据库交互的聊天应用程序，特别演示了在包含花名册的 SQLite 数据库上的应用。
[Semi_Structured_RAG.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/Semi_Structured_RAG.ipynb) | 对包含半结构化数据(文本和表格)的文档执行检索增强生成(RAG)，使用 unstructured 进行解析，multi-vector retriever 进行存储，以及 LCEL 实现链式处理。
[Semi_structured_and_multi_moda...](https://github.com/langchain-ai/langchain/tree/master/cookbook/Semi_structured_and_multi_modal_RAG.ipynb) | 对包含半结构化数据和图像的文档执行检索增强生成(RAG)，使用 unstructured 进行解析，multi-vector retriever 进行存储和检索，以及 LCEL 实现链式处理。
[Semi_structured_multi_modal_RA...](https://github.com/langchain-ai/langchain/tree/master/cookbook/Semi_structured_multi_modal_RAG_LLaMA2.ipynb) | 对包含半结构化数据和图像的文档执行检索增强生成(RAG)，使用各种工具和方法，如 unstructured 进行解析，multi-vector retriever 进行存储，LCEL 实现链式处理，以及开源语言模型如 llama2、llava 和 gpt4all。
[amazon_personalize_how_to.ipynb](https://github.com/langchain-ai/langchain/blob/master/cookbook/amazon_personalize_how_to.ipynb) | 从 Amazon Personalize 获取个性化推荐并使用自定义代理构建生成式 AI 应用
[analyze_document.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/analyze_document.ipynb) | 分析单个长文档。
[autogpt/autogpt.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/autogpt/autogpt.ipynb) | 使用 LangChain 原语(如 LLMs、PromptTemplates、VectorStores、Embeddings 和工具)实现 AutoGPT。
[autogpt/marathon_times.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/autogpt/marathon_times.ipynb) | 实现用于查找马拉松获胜时间的 AutoGPT。
[baby_agi.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/baby_agi.ipynb) | 实现 BabyAGI，一个可以根据给定目标生成和执行任务的 AI 代理，具有更换特定向量存储/模型提供者的灵活性。
[baby_agi_with_agent.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/baby_agi_with_agent.ipynb) | 将 BabyAGI 笔记本中的执行链替换为具有工具访问权限的代理，旨在获取更可靠的信息。
[camel_role_playing.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/camel_role_playing.ipynb) | 实现 CAMEL 框架，用于在大规模语言模型中创建自主合作代理，使用角色扮演和启发式提示来引导聊天代理完成任务。
[causal_program_aided_language_...](https://github.com/langchain-ai/langchain/tree/master/cookbook/causal_program_aided_language_model.ipynb) | 实现因果程序辅助语言链(CPAL)，它通过引入因果结构来改进程序辅助语言(PAL)，以防止语言模型在处理复杂叙述和具有嵌套依赖关系的数学问题时出现幻觉。
[code-analysis-deeplake.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/code-analysis-deeplake.ipynb) | 使用 GPT 和 ActiveLoop 的 Deep Lake 分析其自身代码库。
[custom_agent_with_plugin_retri...](https://github.com/langchain-ai/langchain/tree/master/cookbook/custom_agent_with_plugin_retrieval.ipynb) | 构建一个自定义代理，可以通过检索工具与 AI 插件交互，并围绕 OpenAPI 端点创建自然语言包装器。
[custom_agent_with_plugin_retri...](https://github.com/langchain-ai/langchain/tree/master/cookbook/custom_agent_with_plugin_retrieval_using_plugnplai.ipynb) | 构建一个具有插件检索功能的自定义代理，利用 `plugnplai` 目录中的 AI 插件。
[deeplake_semantic_search_over_...](https://github.com/langchain-ai/langchain/tree/master/cookbook/deeplake_semantic_search_over_chat.ipynb) | 使用 ActiveLoop 的 Deep Lake 和 GPT4 对群聊进行语义搜索和问答。
[elasticsearch_db_qa.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/elasticsearch_db_qa.ipynb) | 使用 Elasticsearch DSL API 与 Elasticsearch 分析数据库进行自然语言交互并构建搜索查询。
[extraction_openai_tools.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/extraction_openai_tools.ipynb) | 使用 OpenAI 工具进行结构化数据提取。
[forward_looking_retrieval_augm...](https://github.com/langchain-ai/langchain/tree/master/cookbook/forward_looking_retrieval_augmented_generation.ipynb) | 实现前瞻性主动检索增强生成(FLARE)方法，该方法生成问题答案，识别不确定的标记，基于这些标记生成假设问题，并检索相关文档以继续生成答案。
[generative_agents_interactive_...](https://github.com/langchain-ai/langchain/tree/master/cookbook/generative_agents_interactive_simulacra_of_human_behavior.ipynb) | 实现一个基于研究论文的生成代理，用于模拟人类行为，使用基于 LangChain 检索器的时间加权记忆对象。
[gymnasium_agent_simulation.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/gymnasium_agent_simulation.ipynb) | 在 Gymnasium 等模拟环境中创建一个简单的代理-环境交互循环，例如基于文本的游戏。
[hugginggpt.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/hugginggpt.ipynb) | 实现 HuggingGPT，一个将 ChatGPT 等语言模型与 Hugging Face 机器学习社区连接的系统。
[hypothetical_document_embeddin...](https://github.com/langchain-ai/langchain/tree/master/cookbook/hypothetical_document_embeddings.ipynb) | 使用假设文档嵌入(HyDE)改进文档索引，这是一种生成和嵌入查询假设答案的嵌入技术。
[learned_prompt_optimization.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/learned_prompt_optimization.ipynb) | 使用强化学习自动增强语言模型提示，可以根据用户偏好个性化响应。
[llm_bash.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/llm_bash.ipynb) | 使用语言学习模型(LLMs)和 Bash 进程执行简单的文件系统命令。
[llm_checker.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/llm_checker.ipynb) | 使用 LLMCheckerChain 函数创建一个自检链。
[llm_math.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/llm_math.ipynb) | 使用语言模型和 Python REPLs 解决复杂的文字数学问题。
[llm_summarization_checker.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/llm_summarization_checker.ipynb) | 检查文本摘要的准确性，并可选择多次运行检查器以获得更好的结果。
[llm_symbolic_math.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/llm_symbolic_math.ipynb) | 使用 LLMs(语言学习模型)和 SymPy(一个用于符号数学的 Python 库)解决代数方程。
[meta_prompt.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/meta_prompt.ipynb) | 实现元提示概念，这是一种构建自我改进代理的方法，代理可以反思自己的表现并相应地修改指令。
[multi_modal_output_agent.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/multi_modal_output_agent.ipynb) | 生成多模态输出，特别是图像和文本。
[multi_modal_RAG_vdms.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/multi_modal_RAG_vdms.ipynb) | 对包含文本和图像的文档执行检索增强生成(RAG)，使用 unstructured 进行解析，Intel 的 Visual Data Management System (VDMS) 作为向量存储，以及链式处理。
[multi_player_dnd.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/multi_player_dnd.ipynb) | 模拟多人地下城与龙游戏，使用自定义函数确定代理的发言顺序。
[multiagent_authoritarian.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/multiagent_authoritarian.ipynb) | 实现一个多代理模拟，其中一个特权代理控制对话，包括决定谁发言以及何时结束对话，背景为模拟新闻网络。
[multiagent_bidding.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/multiagent_bidding.ipynb) | 实现一个多代理模拟，其中代理竞标发言，最高出价者下一个发言，通过虚构的总统辩论示例演示。
[myscale_vector_sql.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/myscale_vector_sql.ipynb) | 访问和交互 MyScale 集成向量数据库，可以增强语言模型(LLM)应用程序的性能。
[openai_functions_retrieval_qa....](https://github.com/langchain-ai/langchain/tree/master/cookbook/openai_functions_retrieval_qa.ipynb) | 通过将 OpenAI 函数集成到检索管道中，在问答系统中构建结构化响应输出。
[openai_v1_cookbook.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/openai_v1_cookbook.ipynb) | 探索 OpenAI Python 库 V1 版本发布的新功能。
[petting_zoo.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/petting_zoo.ipynb) | 使用 Petting Zoo 库创建具有模拟环境的多代理模拟。
[plan_and_execute_agent.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/plan_and_execute_agent.ipynb) | 创建计划和执行代理，通过使用语言模型(LLM)规划任务并使用单独的代理执行任务来完成目标。
[press_releases.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/press_releases.ipynb) | 检索和查询由 [Kay.ai](https://kay.ai) 提供支持的公司新闻稿数据。
[program_aided_language_model.i...](https://github.com/langchain-ai/langchain/tree/master/cookbook/program_aided_language_model.ipynb) | 按照提供的研究论文实现程序辅助语言模型。
[qa_citations.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/qa_citations.ipynb) | 获取模型引用其来源的不同方法。
[rag_upstage_document_parse_groundedness_check.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/rag_upstage_document_parse_groundedness_check.ipynb) | 使用 Upstage Document Parse 和 Groundedness Check 的端到端 RAG 示例。
[retrieval_in_sql.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/retrieval_in_sql.ipynb) | 使用 PGVector 对 PostgreSQL 数据库执行检索增强生成(RAG)。
[sales_agent_with_context.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/sales_agent_with_context.ipynb) | 实现一个上下文感知的 AI 销售代理 SalesGPT，可以进行自然销售对话，与其他系统交互，并使用产品知识库讨论公司的产品。
[self_query_hotel_search.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/self_query_hotel_search.ipynb) | 使用特定酒店推荐数据集构建具有自查询检索功能的酒店房间搜索功能。
[smart_llm.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/smart_llm.ipynb) | 实现 SmartLLMChain，一个自我批评链，生成多个输出建议，批评它们以找到最佳建议，然后改进它以生成最终输出。
[tree_of_thought.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/tree_of_thought.ipynb) | 使用树状思维技术查询大型语言模型。
[twitter-the-algorithm-analysis...](https://github.com/langchain-ai/langchain/tree/master/cookbook/twitter-the-algorithm-analysis-deeplake.ipynb) | 使用 GPT4 和 ActiveLoop 的 Deep Lake 分析 Twitter 算法的源代码。
[two_agent_debate_tools.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/two_agent_debate_tools.ipynb) | 模拟多代理对话，代理可以利用各种工具。
[two_player_dnd.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/two_player_dnd.ipynb) | 模拟一个两人地下城与龙游戏，其中对话模拟器类用于协调主角和地牢主之间的对话。
[wikibase_agent.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/wikibase_agent.ipynb) | 创建一个简单的 WikiBase 代理，利用 SPARQL 生成，并在 http://wikidata.org 上进行测试。
[oracleai_demo.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/oracleai_demo.ipynb) | 本指南概述了如何利用 Oracle AI 向量搜索与 LangChain 一起构建端到端 RAG 管道，并提供分步示例。该过程包括使用 OracleDocLoader 从各种来源加载文档，在数据库内或外部使用 OracleSummary 进行摘要，并通过 OracleEmbeddings 生成嵌入。它还涵盖了使用 OracleTextSplitter 的高级 Oracle 功能根据特定要求对文档进行分块，最后将这些文档存储并索引到向量存储中以供 OracleVS 查询。
[rag-locally-on-intel-cpu.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/rag-locally-on-intel-cpu.ipynb) | 使用 LangChain 和开源工具对本地下载的开源模型执行检索增强生成(RAG)，并在 Intel Xeon CPU 上执行。我们展示了如何将 RAG 应用于 Llama 2 模型，并使其能够回答与 Intel 2024 年第一季度收益发布相关的查询。
[visual_RAG_vdms.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/visual_RAG_vdms.ipynb) | 使用视频和由开源模型生成的场景描述执行视觉检索增强生成(RAG)。
[contextual_rag.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/contextual_rag.ipynb) | 执行上下文检索增强生成(RAG)，在嵌入之前为每个块添加块特定的解释性上下文。
[rag-agents-locally-on-intel-cpu.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/local_rag_agents_intel_cpu.ipynb) | 使用开源模型在本地构建一个 RAG 代理，该代理通过两条路径之一路由问题以找到答案。代理根据从向量数据库检索到的文档生成答案，或者从网络搜索中检索到的文档生成答案。如果向量数据库缺乏相关信息，代理会选择网络搜索。开源模型用于 LLM 和嵌入，并在 Intel Xeon CPU 上本地执行此管道。