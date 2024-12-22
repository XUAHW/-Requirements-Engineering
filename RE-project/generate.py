import json
import os
from openai import OpenAI
from tqdm import tqdm


client = OpenAI(
    api_key="sk-niFhra72GcRwpCl5bPkZ4bIvlKafWSUEVMOiwOgXoCoo8Trd",
    base_url="https://api.chatanywhere.tech/v1"
)


def generate_answer(query):
    prompt = (
        '你是一个需求工程领域的一位专家，我会给你提供一个文档，请你参考这些内容，从给定文档中提取出系统需求'
    )

    completion = client.chat.completions.create(
        model="gpt-4o-mini",  # 使用适当的 GPT 模型
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": query}
        ]
    )
    return completion.choices[0].message.content

def read_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

if __name__ == "__main__":
    api_url = "https://api.chatanywhere.tech/v1/embeddings"  # 替换为 OpenAI embeddings API URL
    api_key = "sk-niFhra72GcRwpCl5bPkZ4bIvlKafWSUEVMOiwOgXoCoo8Trd"  # 替换为你的 OpenAI API 密钥
    csv_file = 'embeddings.csv'  # 替换为你的包含嵌入的 CSV 文件路径

    query = "简短一点，请你阅读下面的内容，使用基于视角的需求提取方法，从中提取出汽车安全监控系统的需求,下面是文档内容:"

    txt_file = 'document.txt'

    content = read_txt_file(txt_file)

    query = query + content


    # 使用生成模型生成回答
    generated_answer = generate_answer(query)

    print(f"生成的回答：\n{generated_answer}")