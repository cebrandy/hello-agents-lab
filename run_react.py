from llm_client import HelloAgentsLLM
from react_agent import ReActAgent
from search_tool import search
from tool_executor import ToolExecutor


def main():
    tools = ToolExecutor()
    tools.registerTool("Search", "搜索网页，获取最新信息", search)

    agent = ReActAgent(HelloAgentsLLM(), tools, max_steps=3)
    answer = agent.run("请先使用 Search 搜索英伟达官网的 GeForce 系列，再简要回答有哪些系列。")

    print("返回值:", answer)


if __name__ == "__main__":
    main()
