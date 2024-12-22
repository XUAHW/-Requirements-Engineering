import requests
import json
import csv
from tqdm import tqdm


# 封装生成embedding的函数
def generate_embedding(text: str, api_url: str, api_key: str) -> list:
    payload = json.dumps({
        "model": "text-embedding-ada-002",
        "input": text
    })

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }

    # 发送请求到GPT接口
    response = requests.post(api_url, headers=headers, data=payload)
    response.raise_for_status()  # 检查请求是否成功

    # 解析JSON响应
    response_data = response.json()

    # 获取embedding
    embedding = response_data['data'][0]['embedding']

    return embedding


# 处理文本文件并生成每个段落的embedding
def process_text_file(file_path: str, api_url: str, api_key: str) -> list:
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # 使用两个空行分割文本为段落
    paragraphs = text.split('\n\n')

    # 为每个段落生成embedding
    embeddings = []
    for paragraph in tqdm(paragraphs, desc="Processing paragraphs"):
        embedding = generate_embedding(paragraph.strip(), api_url, api_key)
        embeddings.append((paragraph.strip(), embedding))

    return embeddings


# 将嵌入保存到CSV文件
def save_embeddings_to_csv(embeddings: list, output_file: str):
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # 写入标题行
        writer.writerow(['Paragraph', 'Embedding'])

        # 写入每个段落及其对应的嵌入
        for paragraph, embedding in embeddings:
            # 将嵌入转换为字符串
            embedding_str = json.dumps(embedding)  # 转换为JSON字符串格式
            writer.writerow([paragraph, embedding_str])


# 示例：调用API并处理文本
if __name__ == "__main__":
    api_url = "https://api.chatanywhere.tech/v1/embeddings"
    api_key = "sk-niFhra72GcRwpCl5bPkZ4bIvlKafWSUEVMOiwOgXoCoo8Trd"  # 替换为您的实际API密钥
    file_path = 'knowledge.txt'  # 替换为您的文件路径
    output_file = 'embeddings.csv'  # 输出CSV文件路径

    # 获取嵌入
    embeddings = process_text_file(file_path, api_url, api_key)

    # 将嵌入保存到CSV文件
    save_embeddings_to_csv(embeddings, output_file)

    print(f"Generated {len(embeddings)} embeddings.")
    print(f"Embeddings saved to {output_file}.")
