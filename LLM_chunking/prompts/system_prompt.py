'''
System prompt settings and functions for LLM seamntic chunking

Author @ Lin Shi
Github: Slimshilin
Date: 2024/05/01
'''

Chinese_system_prompt = "你是一个专业的语义分割助手。我会给你一些文档，你来负责进行语义切割。你的输出格式应为 \n [序号]:::[原文]|||[切割的语义描述]\n 其中不包含[] \n对于原文的部分，你应当采取 [开始词语]...[结束词语]的格式以节省空间（是的，你应当用在你的回答中使用‘...’），但是我需要你的开始词语和结束词语足够长（3-5个字或词）以便能够精准区分在何处进行切割。以下是我想要你进行语义分割的文本：\n"

English_system_prompt = "You are a professional semantic chunking assistant. I will provide you with some documents, and you will be responsible for semantic chunking. Your output format should be: \n[Number]:::[Original Text]|||[Segmented Semantic Description] \n where '[]' should not be included. \n  For the original text part, you should adopt the format of [Starting Words]...[Ending Words] to save space (yes, you should include '...' in your response), but I need your starting words and ending words to be long enough (3-5 words) to accurately distinguish where the chunking should occur. The following are the texts that I want to to do semantic chunking on: \n"


language_prompt_map = {
    'Chinese': Chinese_system_prompt,
    'English': English_system_prompt
}


def build_prompt_by_file_content(file_content_as_a_string, language='Chinese'):
    system_prompt = language_prompt_map[language]

    if type(file_content_as_a_string) !=str:
        raise ValueError("Input file content should be converted into a single string when building prompts!")

    return system_prompt + str(file_content_as_a_string)


def build_prompt(row, language='Chinese'):
    '''
    Build the prompt of a given row of a dataframe that should include the column "content_to_chunk".
    '''

    return build_prompt_by_file_content(str(row['content_to_chunk']), language=language)