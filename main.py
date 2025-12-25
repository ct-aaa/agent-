from agent import run_agent

if __name__ == "__main__":
    # 问题 1: 计数问题 [cite: 44]
    q1 = "在 dataset_B(真实物体)中，被识别为'鸟'的图片有几张？请把它们都展示出来。"
    print(run_agent(q1))

    q2 = "请从 dataset_C (简笔画) 中找出一张'飞机'，并从 dataset_B (真实物体) 中也找出一张'飞机'。"
    print(run_agent(q2))

    q3 = "dataset_B (真实物体) 里，'动物'（如猫、狗、鸟等）和'交通工具'（如飞机、汽车、卡车等）哪个类别的图片更多？"
    print(run_agent(q3))

    q4 = "请检查 dataset_C (简笔画) 中的所有图片，并告诉我它们的分类结果分别是什么？"
    print(run_agent(q4))

    # 问题 5: 计算总和 [cite: 79]
    q5 = "在 dataset_A(手写数字)中，所有图片代表的数字加起来总和是多少？"
    print(run_agent(q5))