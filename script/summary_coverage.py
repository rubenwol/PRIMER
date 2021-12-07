
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import jsonlines

# take from https://github.com/amazon-research/BartGraphSumm/blob/main/src/graph_construction.py
def parser_oie(result, tokens, offset):
    # Result format is same as IOE extaction output
    # Clean the output
    sample = {}
    start = None
    end = None
    key = None
    for ind, tag in enumerate(result["tags"]):
        if tag == "O":
            if start is not None:
                if end is None:
                    sample[key] = {"span": " ".join(tokens[start:start+1]), "index": [start+offset, start+offset]}
                else:
                    sample[key] = {"span": " ".join(tokens[start:end+1]), "index": [start+offset, end+offset]}
                start = None
                end = None
                key = None
        else:
            if tag[:1] == "B":
                if start is not None:
                    if end is None:
                        sample[key] = {"span": " ".join(tokens[start:start+1]), "index": [start+offset, start+offset]}
                    else:
                        sample[key] = {"span": " ".join(tokens[start:end+1]), "index": [start+offset, end+offset]}
                end = None
                start = ind
                key = tag[2:]
            else:
                end = ind
    return sample

def clean_oie(oie_doc_path):
    """
    clean the oie to have Result format same as IOE extaction output
    """
    with jsonlines.open(oie_doc_path) as reader:
        result_lists = list(reader)
    oie_doc_list = []
    for result_list in result_lists:
        counter = 0
        ioes = []
        for result in result_list:
            tokens = result["words"]
            for r in result["verbs"]:
                ioe_sample = parser_oie(r, tokens, counter)
                # print(ioe_sample)
                # if ioe_sample.get("V") is not None and \
                #         ioe_sample.get("ARG0") is not None and \
                #         ioe_sample.get("ARG1") is not None:
                ioes.append(ioe_sample)
            counter += len(tokens)
        oie_doc_list.append(ioes)
    return oie_doc_list

def cleanOie_to_sent(oie):
    sent = ''
    for key in oie.keys():
        sent += oie[key]['span'] + ' '
    return sent[:-1] if len(sent) != 0 else sent

def is_aligned(sent1, sent2, superpal, tokenizer):
    tokens = tokenizer(sent1 + " </s><s> " + sent2, return_tensors="pt")
    outputs = superpal(tokens['input_ids'])
    logits = outputs.logits
    probs = logits.softmax(dim=-1)
    return probs.argmax()

def main():
    tokenizer = AutoTokenizer.from_pretrained("biu-nlp/superpal")
    superpal = AutoModelForSequenceClassification.from_pretrained("biu-nlp/superpal")
    path_docs = '/home/nlp/wolhanr/data/multi_news/oie/test/document/50.jsonl'
    path_summary = '/home/nlp/wolhanr/data/multi_news/oie/test/summary/50.jsonl'
    oie_doc_list = clean_oie(path_docs)
    oie_sum = clean_oie(path_summary)[0]
    print(oie_sum)
    print('number of oie sum clean')
    print(len(oie_sum))
    coverage = [list() for i in range(len(oie_doc_list))]
    coverage_len = [0] * len(oie_doc_list)
    nb_oie_no_match = 0
    for oie in oie_sum:
        sent_sum = cleanOie_to_sent(oie)
        oie_has_match = False
        for i, oies_doc in enumerate(oie_doc_list):
            for oie_doc in oies_doc:
                sent_doc = cleanOie_to_sent(oie_doc)
                if is_aligned(sent_sum, sent_doc, superpal, tokenizer):
                    oie_has_match = True
                    coverage[i].append({sent_sum, sent_doc})
                    coverage_len[i] += 1
            nb_oie_no_match = nb_oie_no_match+1 if oie_has_match==False else nb_oie_no_match

    print(coverage)
    print(coverage_len)
    print(nb_oie_no_match)

if __name__ == "__main__":
    main()