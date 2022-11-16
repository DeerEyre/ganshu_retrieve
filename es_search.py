from elasticsearch import Elasticsearch
import jieba.posseg as psg
import jieba
import pickle


def query_word_two(text, word_dict, jieba_stop=None):
    """
    自定义的分词方案
    :param text: 需要分词的文本
    :param word_dict: 自定义的词库
    :return:分词后的列表
    """
    text = text.lower().replace(" ", "")
    word_list = list()
    # words, pos = lac.run(text)

    words, pos = [], []
    words_dict_pos = {}
    for v, w in psg.lcut(text):
        words.append(v)
        pos.append(w)
        words_dict_pos[v] = w

    last_st = []
    good_word_ids = []
    for i in range(len(words)):
        lab = False
        if len(good_word_ids) > 0:
            if i <= good_word_ids[-1]:
                continue
        if pos[i] in ["uj", "c", "p"]:
            word_list.append(words[i])
            continue

        if words[i] in word_dict:  # 如果本身在词库，直接添加
            lab = True
            good_word_ids = [i]
        if i < len(words) - 1:  # 往前加一
            if "".join(words[i:i + 2]) in word_dict:
                good_word_ids = [i, i + 1]
                lab = True

        if i < len(words) - 2:  # 往前加二
            if "".join(words[i:i + 3]) in word_dict:
                good_word_ids = [i, i + 1, i + 2]
                lab = True

        if i < len(words) - 3:  # 往前加三
            if "".join(words[i:i + 4]) in word_dict:
                good_word_ids = [i, i + 1, i + 2, i + 3]
                lab = True

        if i < len(words) - 4:  # 往前加4
            if "".join(words[i:i + 5]) in word_dict:
                good_word_ids = [i, i + 1, i + 2, i + 3, i + 4]
                lab = True

        if i < len(words) - 5:  # 往前加5
            if "".join(words[i:i + 6]) in word_dict:
                good_word_ids = [i, i + 1, i + 2, i + 3, i + 4, i + 5]
                lab = True

        if len(good_word_ids) > 0:
            new_word = "".join(words[good_word_ids[0]:good_word_ids[-1] + 1])
            if len(last_st) == 0 or new_word != last_st[-1] or new_word == words[i]:
                word_list.append(new_word)

            last_st.append(new_word)
        if not lab:
            word_list.append(words[i])

    result = {"cutWord": {"word": [], "jieba": []}, "stopWord": []}

    query_word_all = set(word_list + words)

    for i in query_word_all:
        if i in jieba_stop and jieba_stop[i] > 40000:
            result["stopWord"].append(i)

        else:
            if i in words:
                result["cutWord"]["jieba"].append(i)
            else:
                result["cutWord"]["word"].append(i)

    return result


def es_search_title_boost(title, size, new_good_word_dict=None, es=None, jieba_stop=None):
    cut_word = query_word_two(title, new_good_word_dict, jieba_stop=jieba_stop)

    query_cutWord = [{"term": {"cutWord": {"value": i, "boost": 2}}} for i in cut_word["cutWord"]["word"]]
    query_cutWord.extend([{"term": {"cutWord": {"value": i, "boost": 1}}} for i in cut_word["cutWord"]["jieba"]])

    query_stopWord = [{"term": {"stopWord": {"value": i, "boost": len(i)}}} for i in cut_word["stopWord"]]

    query = {"bool": {"must": [
        {"bool": {"should": query_cutWord}},
        {"bool": {"should": query_stopWord}}]}}
    # index_title_687w_v2
    # return es.search(index="index_title_51", query=query, size=size)["hits"]["hits"]
    return es.search(index="index_title_687w_v2", query=query, size=size)["hits"]["hits"]


def es_search_title_main(title, num=3, min_score=30, size=5, new_good_word_dict=None, es=None, jieba_stop=None):
    es_result = es_search_title_boost(title, size, new_good_word_dict=new_good_word_dict, es=es, jieba_stop=jieba_stop)

    if len(es_result) > num and es_result[num - 1]["_score"] >= min_score:
        return es_result

    else:
        es_result_fuzz = es_search_title_fuzziness(title, size, jieba_stop=jieba_stop, es=es)
        es_result_score = dict(zip([i["_id"] for i in es_result], [i["_score"] for i in es_result]))

        es_result.extend(
            [i for i in es_result_fuzz if i["_id"] not in es_result_score or i["_score"] > es_result_score[i["_id"]]])

        es_result = sorted(es_result, key=lambda x: x["_score"], reverse=True)

        return es_result[:len(es_result_fuzz)]


def load_good_word(good_word_path):
    word_dict = {}
    with open(good_word_path, "r", encoding="utf-8") as read:
        raw_data = read.read()
        for line_data in raw_data.split("\n"):
            w2f = line_data.split(" : ")
            try:
                if int(w2f[1]) * len(w2f[0]) > 6:
                    if len(w2f[0]) > 12 and (len(w2f[0]) - 12) * int(w2f[1]) > 6:
                        pass
                    else:
                        word_dict[w2f[0]] = w2f[1]
            except IndexError:
                print(line_data)
    return word_dict


def clear_title(title, jieba_stop=None):
    jieba_title = list(jieba.cut(title))
    new_title, new_title_reverse = [], []

    for n, word in enumerate(jieba_title):
        if word in jieba_stop and jieba_stop[word] > 5000:
            continue
        else:
            new_title = jieba_title[n:]
            break

    for n, word in enumerate(reversed(new_title)):
        if word in jieba_stop and jieba_stop[word] > 5000:
            continue
        else:
            if n == 0:
                new_title_reverse = new_title[:]
            else:
                new_title_reverse = new_title[:-n]
            break

    text = "".join(new_title_reverse)

    if len(text) > 2 and len(text) / len(title) < 0.25:
        return title

    return text


def es_search_title_fuzziness(title, size, jieba_stop=None, es=None):
    query = {"multi_match": {
        "query": clear_title(title, jieba_stop=jieba_stop),
        "fields": ["title"],
        "fuzziness": "AUTO"}}

    #     for data in es.search(index="index_title_51", query=query)["hits"]["hits"]:
    #         print(data["_source"]["title"], data["_score"])

    return es.search(index="index_title_687w_v2", query=query, size=size)["hits"]["hits"]


def get_esObject_jiebaStop_newGoddWordDict(es_address="http://192.168.11.4:32182"):
    es = Elasticsearch(es_address)

    file = open("./jieba_title.pkl", "rb")

    jieba_stop = pickle.load(file)
    jieba_stop = dict(jieba_stop)
    new_good_word_dict = load_good_word("./new_good.txt")
    return es, jieba_stop, new_good_word_dict


if __name__ == '__main__':
    es, jieba_stop, new_good_word_dict = get_esObject_jiebaStop_newGoddWordDict()

    title_list = ["基于科学划界的弦理论分析", "刑事诉讼中电子证据取证研究", "生态补偿的哲学思考—以退耕还林工程为例",
                  "先秦楚地诗歌研究", "以舞蹈为核心的民族美育课程建设研究", "烟酰胺与几种药物的相互作用研究",
                  "芥川龙之介与鲁迅作品比较研究", "思想政治教育视域下大学生创新创业教育研究",
                  "基于遥感的露天灰岩矿山开采信息提取", "基于遗传算法的并行化K-means聚类算法研究",
                  "毛红椿天然林群落土壤丛枝菌根真菌群落特征研究", "新型城镇化视化下安庆市户籍制度改革研究",
                  "鸡病诊断与防控信息系统的研制", "建国以后毛泽东社会主义革命思想探析",
                  "海棠花馆藏江西新出宋元买地券整理与研究", "军民耦合互动对军工企业绩效的影响研究",
                  "开放经济条件下货币政策选择与中国经济波动", "高铁可达性对城市经济增长影响的研究",
                  "基于机动车排放模型的地下车库污染物排放量研究", "四旋翼无人机的室内路径规划技术研究"]

    title_list = ["生态补偿的哲学思考—以退耕还林工程为例", "先秦楚地诗歌研究", "以舞蹈为核心的民族美育课程建设研究", "烟酰胺与几种药物的相互作用研究"]
    for title in title_list:
        print("title:", title)
        es_result_title = es_search_title_main(title, size=20, new_good_word_dict=new_good_word_dict, es=es, jieba_stop=jieba_stop)

        for data in es_result_title:
            #         print(data["_source"]["title"], data["_score"])
            print("\t" + data["_source"]["title"], data["_score"])
        print("=" * 50)
