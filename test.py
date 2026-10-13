from llm_client import HelloAgentsLLM


def main() -> None:
    llm = HelloAgentsLLM()
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that writes Python code.",
        },
        {"role": "user", "content": "写一个快速排序算法"},
    ]

    print("--- 调用 LLM ---")
    response = llm.think(messages)
    print("\n\n--- 完整模型响应 ---")
    print(response)


if __name__ == "__main__":
    main()
