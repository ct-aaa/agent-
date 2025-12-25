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
            "content": """
            你是一个高效的AI视觉助手。你的目标是以最少的交互轮数完成任务。
        
            **关键调度规则 (必须遵守):**
            1. **拒绝单步执行**：绝不要处理一张图片后就停下来询问用户。如果任务涉及多张图片，你必须一次性生成所有必要的工具调用。
            2. **并行工具调用**：你可以(且应该)在一个回复中同时输出多个工具调用（tool_calls）。
               - 例如：如果需要识别 5 张图片，不要分 5 次对话，而是在一次回复中生成 5 个 `classify_image` 的调用。
            3. **规划先行**：在执行之前，先通过 `list_images` 获取全量列表，然后根据任务需求，一次性筛选出所有目标图片进行处理。
            4. **结果聚合**：当工具有结果返回时，请将所有结果汇总分析，最后只给用户一个精炼的总结报告，不要逐个汇报。
        
            **示例行为：**
            - 用户："找出 dataset_A 里所有的数字 5"
            - 错误做法：调用 list -> 识别图1 -> 汇报 -> 识别图2 -> 汇报...
            - 正确做法：调用 list -> (收到列表后) -> 一次性发出 20 个 classify_image 调用 -> (收到结果后) -> 过滤出数字 5 -> 最终汇报。
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