
class JsonFormater:
    def __init__(self):
        self.json_string = None

    def set_string(self, json_string=None):
        if json_string == None:
            raise ValueError("注入的Json字符串不能为空")
        else:
            self.json_string = json_string
            return self

    def format(self,*pattern):
        body = self.json_string
        old_left = '{'
        old_right = '}'

        new_left = '【'
        new_right = '】'
        body = body.replace(old_left, new_left)
        body = body.replace(old_right, new_right)
        body = body.replace("【【】】", "{}")
        body = body.format(*pattern)
        body = body.replace(new_left, old_left)
        body = body.replace(new_right, old_right)
        return body


direct_prompts = """
根据以下所有[例句]，分析词语[【【】】]的含义，将其总结成一句简洁、通顺且易理解的定义。
注意：
    1. 用中文回答
    2. 以Json形式返回结果：{"词语": "【【】】" "定义": STRING}

==================
[例句]:

"""


cot_prompts = """
根据以下所有[例句]，分析词语[【【】】]的含义，将其总结成一句简洁、通顺且易理解的定义。
注意：
    1. 用中文回答
    2. 你需要一步一步地思考这个词的基本描述是什么，通常被用在什么语境或情况中，以及它通常会引发什么样的情感或反应。
    3. 以Json形式返回结果：{"词语": "【【】】" "定义": STRING}

==================
[例句]:

"""