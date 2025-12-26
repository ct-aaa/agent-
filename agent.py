import json
from openai import OpenAI
import tools
import os  # 新增
from dotenv import load_dotenv  # 新增

load_dotenv()
# --- 配置 ---
# 2. 从环境变量读取 Key
api_key = os.getenv("DASHSCOPE_API_KEY")

if not api_key:
    raise ValueError("未找到 API Key，请检查 .env 文件！")

client = OpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# --- 工具定义 (含计算器) ---
tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "list_images",
            "description": "列出指定数据集文件夹下的所有图片路径。",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_name": {"type": "string", "description": "数据集名称"}
                },
                "required": ["dataset_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "classify_image",
            "description": "识别单张图片。只需提供图片路径，函数会自动选择模型。返回结果包含文件名，方便对应。",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string", "description": "图片的完整文件路径"}
                },
                "required": ["image_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "精确计算数学表达式。当需要对多个数字求和、比较大小或统计时，必须使用此工具，严禁心算。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式，如 '1+2+3'"}
                },
                "required": ["expression"]
            }
        }
    }
]


def run_agent(user_question):
    print(f"\n{'=' * 40}\nUser: {user_question}\n{'=' * 40}")

    messages = [
        {
            "role": "system",
            "content": """
            你是一个严谨的AI视觉助手。请遵循以下规则：
            1. **并行识别**：获取图片列表后，请在一个步骤中生成所有图片的 `classify_image` 调用，不要分批。
            2. **依赖计算器**：涉及数字加减乘除（如求总和、比大小），**必须**调用 `calculate` 工具，绝对不要自己心算。
            3. **事实导向**：工具返回了什么就是什么，不要臆造结果。
            """
        },
        {"role": "user", "content": user_question}
    ]

    max_steps = 15
    step = 0

    while step < max_steps:
        step += 1
        try:
            response = client.chat.completions.create(
                model="qwen-plus",
                messages=messages,
                tools=tools_schema,
                tool_choice="auto"
            )
        except Exception as e:
            return f"API Error: {e}"

        msg = response.choices[0].message
        messages.append(msg)

        if msg.tool_calls:
            print(f"Step {step}: Agent 正在调用 {len(msg.tool_calls)} 个工具...")

            for tool_call in msg.tool_calls:
                func_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                tool_result = "Error"

                # === 路由逻辑 ===
                if func_name == "list_images":
                    tool_result = tools.list_images(args['dataset_name'])
                    print(f"  -> list_images: 找到 {len(tool_result)} 张图")

                elif func_name == "classify_image":
                    tool_result = tools.classify_image(args['image_path'])
                    # 可以在这里打印部分结果用于调试，但为了整洁省略

                elif func_name == "calculate":
                    tool_result = tools.calculate(args['expression'])
                    print(f"  -> 计算: {args['expression']} = {tool_result}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(tool_result)
                })
            continue
        else:
            print(f"Step {step}: Agent 回复 -> {msg.content}")
            return msg.content

    return "交互超时。"