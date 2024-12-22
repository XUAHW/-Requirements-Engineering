import requests
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# 生成查询的嵌入向量
def generate_query_embedding(query: str, api_url: str, api_key: str) -> list:
    """
    生成查询文本的嵌入
    :param query: 输入的查询文本
    :param api_url: API URL
    :param api_key: API 密钥
    :return: 查询嵌入向量
    """
    payload = json.dumps({
        "model": "text-embedding-ada-002",
        "input": query
    })

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }

    response = requests.post(api_url, headers=headers, data=payload)
    response.raise_for_status()  # 检查请求是否成功

    response_data = response.json()
    query_embedding = response_data['data'][0]['embedding']
    return query_embedding

# 计算余弦相似度的函数
def compute_cosine_similarity(query_embedding, document_embeddings):
    """
    计算查询嵌入与所有文档嵌入的余弦相似度，并返回最相似的文档
    :param query_embedding: 查询的嵌入向量
    :param document_embeddings: 知识库文档的嵌入向量
    :return: 排序后的相似度和文档索引
    """
    similarities = cosine_similarity([query_embedding], document_embeddings)
    sorted_indices = similarities[0].argsort()[::-1]  # 从高到低排序
    return sorted_indices, similarities[0]

# 检索与查询最相关的段落
def retrieve_similar_paragraphs(query, csv_file, api_url, api_key, top_k=3):
    """
    根据查询与 CSV 文件中的段落进行相似度匹配，返回前 k 个最相关段落的索引
    :param query: 用户查询
    :param csv_file: 包含文档的 CSV 文件路径
    :param api_url: API URL
    :param api_key: API 密钥
    :param top_k: 返回最相关的前 k 个段落
    :return: 前 k 个最相关段落的索引
    """
    # 读取 CSV 文件，假设每个段落存在 'paragraph' 和 'embedding' 列中
    df = pd.read_csv(csv_file)
    document_texts = df['Paragraph'].tolist()  # 获取所有段落文本
    document_embeddings = np.array(df['Embedding'].apply(json.loads).tolist())  # 加载存储的嵌入

    # 为查询生成嵌入
    query_embedding = generate_query_embedding(query, api_url, api_key)

    # 计算查询与文档的相似度，返回最相关的文档
    sorted_indices, similarities = compute_cosine_similarity(query_embedding, document_embeddings)

    # 返回前 top_k 个最相关段落的索引
    return sorted_indices[:top_k], similarities[sorted_indices[:top_k]]


# 示例：调用检索函数
if __name__ == "__main__":
    api_url = "https://api.chatanywhere.tech/v1/embeddings"  # 替换为 OpenAI embeddings API URL
    api_key = "sk-niFhra72GcRwpCl5bPkZ4bIvlKafWSUEVMOiwOgXoCoo8Trd"  # 替换为你的 OpenAI API 密钥
    csv_file = 'embeddings.csv'  # 替换为你的包含嵌入的 CSV 文件路径

    query = "需求获取"

    # 检索与查询最相似的段落
    top_k_indices, similarities = retrieve_similar_paragraphs(query, csv_file, api_url, api_key, top_k=5)

    # 输出最相关的段落索引和相似度
    print(f"最相关的段落索引: {top_k_indices}")
    print(f"相似度: {similarities}")
