# Requirements-Engineering
document.txt -> 需要提取需求的文档
embeddings.csv -> 保存着外部知识库的内容与其对应的文本嵌入
generate.py -> 使用原始的大语言模型进行需求提取
generate_rag.py -> 使用RAG+大语言模型进行需求提取
knowledge.txt -> 保存着外部知识库的内容
knowledge_embedding_generate.py -> 将外部知识库的内容转换为对应的文本嵌入
query_retrieval.py -> 检索模块,用于检索外部知识库中与query最相近的几部分内容
requirements.txt -> 项目所需的依赖
