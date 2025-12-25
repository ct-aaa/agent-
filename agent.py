import json
from openai import OpenAI
import tools  # 导入你的工具模块

# --- 配置 ---
# 请确保 api_key 和 base_url 正确
client = OpenAI(
    api_key="sk-15d3ce6e67504bf994191e6ae9ef6b02",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# --- 定义工具 Schema ---
tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "list_images",
            "description": "列出指定数据集文件夹下的所有图片路径。",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_name": {"type": "string",
                                     "description": "数据集名称, 例如 'dataset_A', 'dataset_B', 'dataset_C'."}
                },
                "required": ["dataset_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "classify_image",
            "description": "识别单张图片的内容类别。",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string", "description": "图片的完整文件路径。"}
                },
                "required": ["image_path"]
            }
        }
    }
]


def run_agent(user_question):
    print(f"\n{'=' * 20}\nUser: {user_question}\n{'=' * 20}")

    # 初始化对话历史
    # 强制 System Prompt 更加严格，防止它瞎编工具名
    messages = [
        {
            "role": "system",
            "content": """你是一个智能体。你可以使用工具来回答问题。
            1. 必须严格使用 'classify_image' 来识别图片，不要编造其他工具名（如 recognize_digit）。
            2. 如果需要处理多张图，请在一个回复中生成多个 tool_call并一次完成处理。
            3. 如果你只得到了图片列表，下一步必须调用 classify_image。
            """
        },
        {
            "role": "user",
            "content": user_question
        }
    ]

    # --- 核心循环：支持多轮对话 ---
    max_steps = 30  # 防止死循环，最多交互15次
    step = 0

    while step < max_steps:
        step += 1

        # 1. 请求 LLM
        try:
            response = client.chat.completions.create(
                model="qwen-plus",
                messages=messages,
                tools=tools_schema,
                tool_choice="auto"
            )
        except Exception as e:
            return f"API 调用出错: {e}"

        msg = response.choices[0].message
        messages.append(msg)  # 把 LLM 的回复加入历史

        # 2. 判断 LLM 的行为

        # 情况 A: LLM 想要调用工具
        if msg.tool_calls:
            print(f"Step {step}: Agent 决定调用 {len(msg.tool_calls)} 个工具...")

            # 执行所有请求的工具
            for tool_call in msg.tool_calls:
                func_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)

                tool_result = "Error: Tool not found"

                # 路由到具体函数
                if func_name == "list_images":
                    tool_result = tools.list_images(args['dataset_name'])
                    print(f"  -> 执行 list_images: 找到 {len(tool_result)} 张图")
                    # 提示：如果图片太多，可以只返回前几张给 LLM 测试，或者全部返回

                elif func_name == "classify_image":
                    tool_result = tools.classify_image(args['image_path'])
                    # print(f"  -> 执行 classify_image: {args['image_path']} -> {tool_result}")

                # 把执行结果封装成 message
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(tool_result)
                })

            # 工具执行完，不 return，直接进入下一次 while 循环，把结果给 LLM 看
            continue

            # 情况 B: LLM 没有调用工具，直接返回了文本
        else:
            # 这是一个简单的启发式判断：
            # 如果 LLM 只是在“自言自语”规划（比如“我接下来要...”），我们可能需要强行提示它继续。
            # 但通常现在的强模型（Qwen/DeepSeek）如果不调用工具且输出了文本，通常意味着它认为任务完成了，或者是单纯的聊天。

            content = msg.content
            print(f"Step {step}: Agent 思考/回复 -> {content}")

            # 如果回复里看起来像是最终答案，或者我们不想让它无限聊下去，就结束
            # 这里简单处理：只要没有 tool_calls，就认为是最终答案并返回
            return content

    return "任务超时，达到最大交互次数。"