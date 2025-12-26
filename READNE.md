项目gtihub仓库：[https://github.com/ct-aaa/agent-](https://github.com/ct-aaa/agent-)

## 1. 系统概览
本项目实现了一个能够“看图说话”并进行逻辑推理的 AI Agent。它不仅仅是一个聊天机器人，更是一个能通过 Python 代码主动操作本地文件、调用深度学习模型并进行数学计算的自动化系统。

核心工作流：

用户提问 (main.py) $ \rightarrow $ Agent 规划任务 (agent.py) $ \rightarrow $调用视觉/计算工具 (tools.py) $ \rightarrow $ 汇总结果反馈用户。

---

## 2. 核心模块分析
### 2.1 控制中心 (`agent.py`)
这是系统的大脑，负责任务规划和工具调度。

+ **ReAct 循环机制**：
    - 使用了 `While` 循环结构，允许 Agent 进行多轮思考（默认最大 15 步）。
    - **思考-行动-观察**：Agent 接收用户指令，决定调用哪个工具（行动），获取工具返回的结果（观察），然后决定下一步操作或回答用户。
+ **工具定义 (Schema)**：
    - 向 LLM 注册了三个核心工具：`list_images`（查文件）、`classify_image`（看图）、`calculate`（计算）。
+ **Prompt 工程 (系统提示词)**：
    - **并行处理**：强制 Agent 一次性列出所有识别任务，避免低效的单张循环。
    - **严谨计算**：强制 Agent 遇到数字问题必须调用 `calculate`，禁止不可靠的心算。
    - **事实导向**：要求 Agent 严格依据工具返回结果作答。
+ **安全配置**：
    - 引入 `python-dotenv` 库，从 `.env` 文件加载 API Key，避免了密钥硬编码的安全风险。

### 2.2 视觉与执行终端 (`tools.py`)
这是系统的四肢，负责具体的任务。

+ **多模型适配与路由 (Model Router)**：
    - 系统集成了三个针对不同数据集的专用模型，并实现了自动路由逻辑：
        * **Dataset A (MNIST)**: 使用自定义的 `BetterCNN` 轻量级卷积网络。
        * **Dataset B (CIFAR-10)**: 动态加载 `ResNet20` (优先本地缓存，兜底远程下载)。
        * **Dataset C (Sketch/TU-Berlin)**: 使用微调过的 `ResNet50`，支持加载自定义权重文件。
+ **单例模式 (Singleton Pattern)**：
    - 实现了 `_MODELS` 缓存机制。模型只在第一次被调用时加载到显存/内存中，后续调用直接复用，极大提升了运行效率。
+ **精细化图像预处理**：
    - 针对不同数据集做了特定的 Transform：
        * **A**: 转灰度，标准化。
        * **C (简笔画)**: 关键步骤是 **颜色反转 (Invert)** 和 **二值化**。因为通常模型训练用的是白底黑线条，而输入可能是黑底白线条，或者为了增强特征提取。
+ **结果绑定**：
    - `classify_image` 返回的字符串格式为 `"[文件名] 的识别结果是: 类别"`。这种显式的绑定防止 LLM 在处理多张图片结果时发生“张冠李戴”的幻觉。
+ **增强型计算器**：
    - `calculate` 函数不仅支持基础四则运算，还扩展支持了逻辑比较符（`>`, `<`）和取模运算（`%`），这使得 Agent 可以回答如“图片数量是否多于 X”或“奇偶数判断”等逻辑问题。

### 2.3 测试入口 (`main.py`)
+ 包含了一系列测试用例（Q1 - Q8），覆盖了计数、跨数据集检索、逻辑比对、总和计算等场景，验证了 Agent 的多步推理能力。

### 2.4 模型生成方法（`model_train_tools`）
+ 包含了项目A，C的数据集，以及简笔画分类任务C的`model_best_TUBerlin.pth`的练模型，以及A,C的训练函数方法。</font>

---

## 3.项目实现效果说明
### 3.1 模型工具分类能力
+ 手写数字

<!-- 这是一张图片，ocr 内容为：LILLIIII 91 [11.PNG]的识别结果是:6 11.PNG 16 的识别结果是:6 [12.PNG] 12.PNG 1 5 的识别结果是:5 [13.PNG] 13.PNG 17 的识别结果是:7 [14.PNG] 1 14.PNG 18 的识别结果是:8 [15.PNG] 1 15.PNG 的识别结果是:1 1 1 [16.PNG] 16.PNG 一0 [17.PNG] 的识别结果是:0 17.PNG 1 1 的识别结果是:1 18.PNG [18.PNG] 9_ [19.PNG] 的识别结果是:6 19.PNG [2.PNG] 4 的识别结果是:4 2.PNG 14 [20.PNG] 的识别结果是:4 20.PNG 1 7 [3.PNG] 的识别结果是:7 3.PNG 9L 1 4.PNG [4.PNG] 的识别结果是:6 [5.PNG] 1 7 1 5.PNG 的识别结果是:7 61 [6.PNG] 1 6.PNG 的识别结果是:9 [7.PNG]的识别结果是:0 0 1 7.PNG I 5 [8.PNG]的识别结果是:5 18.PNG 1 9.PNG [9.PNG] 1 8 的识别结果是:8 准确率:100.00%(20/20) -->
![](https://cdn.nlark.com/yuque/0/2025/png/43187566/1766743057898-0573ae37-ad57-4e9a-9db9-e23cb76f0f6c.png)

+ 彩图识别

<!-- 这是一张图片，ocr 内容为：|[13.PNG] 的识别结果是: HORSE HORSE 13.PNG 个小与平台面 [14.PNG]的识别结果是:HORSE 14.PNG HORSE [15.PNG]的识别结果是:HORSE 15.PNG HORSE I PLANE 16.PNG [[16.PNG]的识别结果是:PLANE [17.PNG]的识别结果是:PLANE 2/20 [00:00<00:00, 19.48IT/S]L 17.PNG 10% PLANE [[18.PNG]的识别结果是:SHIP 1 18.PNG SHIP | [19.PNG] 的识别结果是:SHIP 1 19.PNG SHIP | [2.PNG]的识别结果是:BIRD 12.PNG BIRD [20.PNG] 的识别结果是:SHIP 20.PNG SHIP [[3.PNG]的识别结果是:CAR 3.PNG CAR 14.PNG |[4.PNG] 的识别结果是:CAR CAR 1 5.PNG 1[5.PNG] 的识别结果是:CAR CAR 6.PNG [6.PNG] 的识别结果是:CAT CAT 1 7.PNG 1 [7.PNG] G]的识别结果是:CAT CAT 18.PNG I [8.PNG]的识别结果是:DEER DEER 19.PNG [9.PNG]的识别结果是:DEER DEER 准确率:100.00%(20/20) -->
![](https://cdn.nlark.com/yuque/0/2025/png/43187566/1766743080794-ccb99c2c-3df0-4763-af60-72abed80ba83.png)

+ 简笔画

<!-- 这是一张图片，ocr 内容为：[3178.PNG] 的识别结果是: CALCULATOR 3178.PNG CALCULATOR 3410.PNG [3410.PNG] 的识别结果是:CANDLE CANDLE 3448.PNG [3448.PNG]的识别结果是: CANNON CANNON 5%11 [4754.PNG]的识别结果是: [00:00<00:11,  1.66IT/S]L 4754.PNG 1/20 COW COW [8284.PNG] 的识别结果是:HEDGEHOG 8284.PNG HEDGEHOG | [8411.PNG]的识别结果是:HELMET 8411.PNG HELMET [8550.PNG]的识别结果是:HORSE 8550.PNG HORSE ICE-CREA.. [9016.PNG] 的识别结果是:ICE-[ V 9016.PNG ICE-CREAM-CONE [12092.PNG] 的识别结果是:PERSON SITTING 12092.PNG PERSON [00:00<00:00, 26.00IT/S]L 13096.PNG 1 14/20 70% [13096.PNG] 的识别结果是:PRETZ I PRETZEL [13873.PNG] 的识别结果是: ROOSTER 13873.PNG ROOSTER [14435.PNG]的识别结果是:SCORPION 14435.PNG SCORPION [15840.PNG] 的识别结果是:SPACE SHUTTLE 15840.PNG SPACE SH.. [16118.PNG]的识别结果是:SPOON 16118.PNG SPOON [17104.PNG] 的识别结果是:SYRINGE 17104.PNG SYRINGE [17680.PNG] 的识别结果是:TELEPHONE 17680.PNG TELEPHONE [18185.PNG] 的识别结果是:TOOTH 18185.PNG TOOTH 准确率:100.00%(20/20) -->
![](https://cdn.nlark.com/yuque/0/2025/png/43187566/1766743091655-5c9ded90-bb53-41e3-9766-321895f3b683.png)

### 3.2 模型对话能力
<!-- 这是一张图片，ocr 内容为：AD AGENT DESIGN X 0 疗 M?人工智能原理与技术课程实验.MD TEST ACCURACY PY 项目 GITIGNORE AGENTPY MAIN.PY GITATTRIBUTES PKD_DEPACKE PY DATASET 8 # PRINTURON_AGENT(Q577 AGENT DESIGN EAHOMEWORK\ANAGENT DESIGN 13 口NENV LIBRARY根目录 14 中Q4三"请检查 DATASET.C(简笔画)中的所有图片,并告诉我它们的分类结果分别是什么?" 开品 DATA CACHE DATASETS 15 # PRINT(RUN_AGENT(Q4)) MODEL TRAIN TOOLS 16 17 #问题5:计算总和[CITE:100] PRE TRAIN MODEL BEST TUBERLIN.PTH 18 Q5:"在DATASET_A(手写数字)中,所有图片代表的数字加起来总和是多少?" TEST TOOLS 19 PRINT(RUN_AGENT(Q5)) INIT MODELS LABELS.PY 20 MODEL A MNISTH PY 21 MODEL B.PTHPY # GS 三"请从 DATASET.A画中找出一条代表数字(7"的图片,并在DATASET.8中我到一架马的图片,将他们一间表示. MODEL CPY 22 # PRINT(RUN_AGENT(Q6)) 运行 USER:在DATASET_A(手写数字)中,所有图片代表的数字加起来总和是多少? 创响自 STEP 1:AGENT 正在调用 1 个工具... LIST_IMAGES:找到20张图 STEP 2:AGENT 正在调用 20 个工具... STEP 3:AGENT 正在调用 1 个工具... 计算:5+5+6+6+5+7+8+1+0+0+1+6+4+4+7+6+7+9+0+5+8 ; 100 STEP 4: AGENT 回复 -> R 回复 ->在DATASET_A中,所有图片代表的数字加起来总和是 100. 所有图片代表的数字加起来总和是100. 在 DATASET_A中,所有 O 进程已结束,退出代码为0 9 185(69字符,1行接行彻) CRUF UTF-8 4个空格 PYTHON3.10(AGENT DESIGN) @ O AGENT DESIGN MAINPY -->
![](https://cdn.nlark.com/yuque/0/2025/png/43187566/1766743152149-4d4f7a82-fecd-4416-b7cc-e2ddb114afe5.png)

