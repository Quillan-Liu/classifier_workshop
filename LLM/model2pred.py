import os
import json
import time
# 导入LangChain的ChatOpenAI模块，用于与大语言模型交互
from langchain_openai import ChatOpenAI
# 导入tenacity库的相关装饰器，用于实现重试机制
from tenacity import retry, stop_after_attempt, wait_fixed
import warnings

warnings.filterwarnings("ignore")

# 1.创建大模型客户端实例
llm = ChatOpenAI(base_url=os.getenv('DASHSCOPE_BASE_URL'),
                 api_key=os.getenv('DASHSCOPE_API_KEY'),
                 model='qwen4-max',
                 model_kwargs={"response_format": {"type": "json_object"}})


# 2.重试机制实现大模型多次调用
# stop_after_attempt(3): 最多重试3次
# wait_fixed(2): 每次重试之间等待2秒
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def llm_invoke(llm, prompt):
    """
    重试机制调用大模型
    :param llm: 大模型客户端
    :param prompt: 提示词
    :return: 调用结果
    """
    # invoke(): langchain_openai 的 invoke 方法
    return llm.invoke(prompt)


# 3.定义模型预测函数
def model2pred(title):
    """
    使用大语言模型对新闻标题进行分类预测
    :param title: 待分类的新闻标题
    :return:  分类结果和原因的字典，格式为{"category": "类别", "reason": "分类原因"}  调用时间ms
    """
    # 3.1 统计当前时间
    start_time = time.time()
    # 3.2 编写提示词
    messages = [
        {
            "role": "system",
            "content": """
                        你是一名新闻分类审核员，任务是将新闻标题分类到以下类别之一：
                        finance, realty, stocks, education, science, society, politics, sports, game, entertainment。
                        请根据标题内容和以下关键词与示例，匹配最相关的类别。如果标题涉及教育机构但核心是社会贡献，优先归为 society。
                        请始终以json格式输出：{"category": "类别", "reason": "分类原因"}

                        类别关键词与示例：
                        - finance: 银行、信用卡、贷款、利率 (例: "各银行信用卡挂失费迥异")
                        - realty: 房产、地价、楼盘 (例: "东5环海棠公社230平准现房")
                        - stocks: 股市、股指、期货 (例: "金证顾问：过山车行情意味着什么")
                        - education: 学校、考试、招生 (例: "中华女子学院仅1专业招男生")
                        - science: 技术、网站、宇航 (例: "“手机钱包”亮相科博会")
                        - society: 社会事件、犯罪、公益 (例: "82岁老太为学生做饭扫地44年")
                        - politics: 政策、国际关系 (例: "查韦斯称愿为俄罗斯提供空军基地")
                        - sports: 比赛、运动员、奥运 (例: "卡佩罗：德国脚生猛的原因")
                        - game: 电子游戏、网游、电竞 (例: "《赤壁OL》攻城战硝烟又起")
                        - entertainment: 明星、影视、综艺 (例: "冯德伦徐若瑄隔空传情")
                        """
        },
        {
            "role": "user",
            "content": f"新闻标题：'{title}'，请分类并说明原因。请用json格式回答，包含category和reason字段"
        }
    ]
    # 3.3 调用llm_invoke函数, 返回结果
    response = llm_invoke(llm, messages)
    # print('response--->\n', type(response), response)
    # 3.4 对结果进行解析, 转换python字典数据类型
    data_dict = json.loads(response.content)
    # 3.5 获取字典中的类别 以及 分类原因
    category = data_dict.get("category", "society")
    reason = data_dict.get("reason", "未明确分类，归为社会类别")
    elapsed_time = (time.time() - start_time) * 1000
    # 3.6 返回结果 分类结果和原因的字典 以及 调用时间ms
    return {"category": category, "reason": reason}, elapsed_time


if __name__ == '__main__':
    title = "BLG VS GENG 3:1, BLG夺得了英雄联盟s16世界冠军, bin终于成为世一上!"
    result, duration = model2pred(title)
    print('result--->', result)
    print('duration--->', duration)
