import jieba
from jieba.analyse import tfidf

# 读取文本
with open('data/task5/task5data.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 分句处理
sentences = [s.strip() for s in text.split('。') if len(s) > 5]

# 计算句子重要度
sentence_scores = []
for sent in sentences:
    keywords = tfidf(sent, withWeight=True)
    score = sum([kw[1] for kw in keywords])
    sentence_scores.append((sent, score))

# 去重并排序
unique_sentences = {}
for sent, score in sentence_scores:
    if sent not in unique_sentences or score > unique_sentences[sent]:
        unique_sentences[sent] = score

# 取重要度最高的三句
sorted_sents = sorted(unique_sentences.items(), key=lambda x: x[1], reverse=True)[:3]

# 输出结果
with open('task5result.txt', 'w', encoding='utf-8') as f:
    for sent, _ in sorted_sents:
        f.write(sent + '。\n')