import requests
from time import time, sleep
import json
import os
import concurrent.futures
from tqdm import tqdm
import re

data_compile = re.compile(r"data:(.*)")


url = "http://36.103.203.209:39256"


def login():
    url_use = f"{url}/api/user_manage/users/login"
    data = {
        "username": "admin",
        "password": "admin"
    }

    rsp = requests.post(url_use, data=data)
    print(rsp.json())
    return rsp


def create_project(project_name):
    url_use = f"{url}/api/project_manage/projects/create_project"
    data = {
        "project_name": project_name,
        "params": {},
        "company": "",
        "project_administrator": ["admin"]
    }
    return requests.post(url_use, json=data, headers=headers)


def get_project_id(project_name):
    url_use = f"{url}/api/project_manage/projects/get_project"
    data = {
        "project_name": project_name
    }
    print("111", data)
    return requests.get(url_use, params=data, headers=headers)


def upload_data(pid, data_path):
    url_use = f"{url}/api/knowledge_manage/knowledge/excel_up_data"
    data = {
        "pid": pid,
        "auto_similar_knowledge": 0
    }
    print(data)
    file_name = os.path.basename(data_path)
    with open(data_path, 'rb') as f:
        files = {'file': (file_name, f)}
        print(files)
        return requests.post(url_use, files=files, params=data, headers=headers)


def get_statistics_status(pid):
    url_use = f"{url}/api/knowledge_manage/knowledge/statistics_status"
    data = {
        "pid": pid
    }
    return requests.get(url_use, params=data, headers=headers)


def online_all(pid):
    url_use = f"{url}/api/knowledge_manage/knowledge/online"
    data = {
        "pid": pid,
        "id_list": []
    }
    return requests.post(url_use, json=data, headers=headers)


def multi_thread_excute(all_tasks, parralle_num=5, show_progress_bar=False):
    '''
    多线程运行任务，注意，返回结果序并不和all_tasks一致，请设计好task的输出，能够通过map的形式找到对应的答案
    '''
    def multi_thread_excute_helper(tasks):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            exe_tasks = [executor.submit(*task) for task in tasks]
            results = [future.result()
                       for future in concurrent.futures.as_completed(exe_tasks)]
        return results

    all_results = []
    for i in tqdm(range(0, len(all_tasks), parralle_num), disable=not show_progress_bar):
        all_results += multi_thread_excute_helper(
            all_tasks[i:i + parralle_num])
        # sleep(0.33)
    return all_results


def search_concurrency_test(project_name, num=10):
    url_use = f"{url}/api/search_api/search/search"

    headers = {
        'Content-Type': 'application/json'
    }

    import requests

    def send_search_request(query, i):
        data = {
            "query": query,
            "project_name": project_name,
            "uuid": str(i)
        }
        start = time()
        rsp = requests.post(url_use, json=data, headers=headers)
        assert len(rsp.json()["data"]) > 0, "无搜索结果"
        cost = time() - start
        # print("time costs", cost)
        return cost

    from time import time

    all_tasks = []
    for _ in range(4):
        for i, query in enumerate(open("./dataset/nl_queries.txt").readlines()):
            query = query.strip()
            all_tasks.append([send_search_request, query, i])
    all_cost = multi_thread_excute(all_tasks, parralle_num=num)
    import pandas as pd
    print(f"{project_name} 并发 {num} 测试结果: ")
    print(pd.Series(all_cost).describe())
    json.dump(all_cost, open(f"{project_name}_{num}.json",
              "w", encoding="utf-8"), ensure_ascii=False, indent=2)


def embedding_concurrency_test(p, num):
    url_use = f"{url}/api/nlp_service/embedding"

    headers = {
        'Content-Type': 'application/json'
    }

    import requests

    def send_search_request(query, i):
        data = {
            "text": query
        }
        start = time()
        rsp = requests.post(url_use, json=data, headers=headers)
        print(rsp.json())
        cost = time() - start
        # print("time costs", cost)
        return cost

    from time import time

    all_tasks = []
    for i, query in enumerate(open("./dataset/nl_queries.txt").readlines()):
        query = query.strip()
        all_tasks.append([send_search_request, query, i])
    all_cost = multi_thread_excute(all_tasks, parralle_num=num)
    json.dump(all_cost, open(f"{project_name}_{num}_emb.json",
              "w", encoding="utf-8"), ensure_ascii=False, indent=2)


def get_status(pid):
    url_use = f"{url}/api/knowledge_manage/knowledge/get_status"
    data = {
        "pid": pid
    }
    rsp = requests.get(url_use, params=data, headers=headers)
    return all([not status for status in rsp.json()["data"]["button_status"].values()])


def prepare_data():
    project_name = "test"
    print("创建项目...")
    print(create_project(project_name).json())

    project_id = get_project_id(project_name).json()["data"][0]["id"]
    print("准备上传一万条样例数据")
    rsp = upload_data(
        project_id, data_path="./dataset/klg_base_1w.xlsx").json()
    print(rsp)
    assert rsp["message"] == "完成", "上传失败"
    print("请检查后端日志确认")

    # from time import time
    while not get_status(project_id):
        sleep(0.5)

    print("准备上线数据")
    rsp_online = online_all(project_id).json()
    print(rsp_online)
    assert rsp_online["message"] == "完成", "上线失败"
    while not get_status(project_id):
        sleep(0.5)
    print("上线完成")


def get_timestamp():
    return int(time() * 10000)


def decode_bytes(bytes, query):
    all_events = bytes.decode("utf-8").split("\r\n\r\n")
    rsp = []
    for event in all_events:
        if not event.strip():
            continue
        data = data_compile.findall(event)
        if len(data) == 0:
            print("error", event)
            continue
        else:
            data = data[0]
        data_dic = json.loads(data)
        rsp.append(data_dic["content"])

        if "search_result" in data_dic:
            search_result = data_dic["search_result"]

    # prompt的长度为 template长度 + query + reference的长度
    prompt_len = 328 + \
        len(query) + sum([len(i["title"]) +
                          len(i["content"]) + 8 for i in search_result])
    prompt_token_len = prompt_len / 1.6

    # 输出长度
    output_token_len = sum([len(i) for i in rsp]) / 1.6
    after_first_output_token_len = sum([len(i) for i in rsp[1:]]) / 1.6

    # prompt长度， 输出总长度
    return prompt_token_len, output_token_len, after_first_output_token_len


def send_chat_request(pid, query):
    url_use = f"{url}/api/c_qa/c_knowledge_qa/chat"
    data = {
        "query": query,
        "pid": pid,
        "session_time": get_timestamp()
    }
    start = time()
    first_time = None
    rsp = requests.post(url_use, json=data, headers=headers, stream=True)
    bytes = b""
    for a in rsp:
        if first_time is None:
            first_time = time() - start
            decode_start = time()
        bytes += a
        # search_results = json.loads(a.decode("utf-8").split("\r\n")[1])["search_result"]
        # total_tokens_len = sum([len(i["title"] + len(i["content"])) for i in search_results]) / 1.6
        # print("prompt长度: ", total_tokens_len)
    end_time = time()

    cost = end_time - start
    decode_cost = end_time - decode_start

    prompt_token_len, output_token_len, after_first_output_token_len = decode_bytes(
        bytes, query)

    decode_speed = after_first_output_token_len / decode_cost
    overall_speed = (prompt_token_len + output_token_len) / cost

    # 返回prompt_token_len, first_time, decode_speed, overall_speed
    return {
        "prompt_token_len": prompt_token_len,
        "output_token_len": output_token_len,
        "first_time": first_time,
        "after_first_output_token_len": after_first_output_token_len,
        "decode_speed": decode_speed,
        "overall_speed": overall_speed
    }


def chat_concurrency_test(project_name, pid, num):
    all_tasks = []
    for _ in range(1):
        for i, query in tqdm(enumerate(open("./dataset/nl_queries.txt").readlines())):
            query = query.strip()
            all_tasks.append([send_chat_request, pid, query])
    all_cost = multi_thread_excute(
        all_tasks, parralle_num=num, show_progress_bar=True)
    import pandas as pd
    print(f"{project_name} 并发 {num} 测试结果: ")
    df = pd.DataFrame(all_cost)
    df["bin"] = pd.qcut(df["prompt_token_len"], q=5)
    all_bins_idx = sorted(df["bin"].unique(), key=lambda x: x.left)
    for bin in all_bins_idx:
        print(f"bin in {bin}:")
        print(df[df["bin"] == bin].describe())
    df.to_json(f"{project_name}_{num}.json", force_ascii=False)
    # json.dump(all_cost, open(f"{project_name}_{num}.json",
    #           "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    token = login().json()["data"]["hash_token"]
    headers = {
        'Authorization': f'Bearer {token}',
    }
    # print(send_chat_request(1, "聊个天"))
    # print(chat_concurrency_test("new_traffic_management_test", 1, 3))
    prepare_data()
    for project_name in ["test"][:1]:
        for num in [1, 3, 5][:]:
            chat_concurrency_test(project_name, 2, num=num)