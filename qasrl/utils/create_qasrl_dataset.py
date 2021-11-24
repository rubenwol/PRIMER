import sys
import os
import json
import jsonlines
import argparse

from datasets import load_dataset, load_metric, DatasetDict


def choose_answer(spans):
    answer = ''
    if len(spans) == 1:
        return spans[0]['text']
    for span in spans:
        if issubspan(span, spans):
            continue
        else:
            answer += '<a> ' + span['text'] + ' '
    # delete the last space
    return answer[:-1]


def list_qa_pairs(args, hf_datasets):
    '''
    create for train , validation and test set a list of qasrl document according to the source set
    '''
    output = [None] * 3
    output[0] = [''] * hf_datasets['train'].shape[0]
    output[1] = [''] * hf_datasets['validation'].shape[0]
    output[2] = [''] * hf_datasets['test'].shape[0]
    set_index = {'qasrl_train.jsonl': 0,  'qasrl_val.jsonl': 1, 'qasrl_test.jsonl': 2}

    # on multi news list[i] will contains sequences of the qasrl pairs of each of the document separate by the string "|||||"
    if args.dataset_name == 'multi_news':
        d = {}
        for set in os.listdir(args.qasrl_path):
            with jsonlines.open(os.path.join(args.qasrl_path,set), 'r') as reader:
                qasrl_data = list(reader)
            for i in range(len(qasrl_data)):
                topic_id = int(qasrl_data[i]['topic_id'])
                if topic_id not in d.keys():
                    d[topic_id] = dict()
                story_id = qasrl_data[i]['story_id']
                if 'verbs' not in qasrl_data[i].keys():
                    print(qasrl_data[i])
                    continue
                for verb in qasrl_data[i]['verbs']:
                    for qa_pair in verb['qa_pairs']:
                        question = qa_pair['question']
                        answer = choose_answer(qa_pair['spans'])
                        if story_id not in d[topic_id].keys():
                            d[topic_id][story_id] = '<qa/> ' + question + ' ' + answer + ' </qa>'
                        else:
                            d[topic_id][story_id] += ' <qa/> ' + question + ' ' + answer + ' </qa> '

            for i in range(len(output[set_index[set]])):
                output[set_index[set]][i] = '|||||'.join(d[i].values())

    return output


def issubspan(span, spans):
    for s in spans:
        if s == span:
            continue
        if span['start'] >= s['start'] and span['end'] <= s['end']:
            return True
    return False


def main(args):
    hf_datasets = load_dataset(args.dataset_name, cache_dir=args.dataset_cache_dir)
    train_qasrl, val_qasrl, test_qasrl = list_qa_pairs(args, hf_datasets)
    train_dataset = hf_datasets['train'].add_column('qasrl', train_qasrl)
    val_dataset = hf_datasets['validation'].add_column('qasrl', val_qasrl)
    test_dataset = hf_datasets['test'].add_column('qasrl', test_qasrl)
    hf_datasets = DatasetDict({'train': train_dataset,
                               'validation': val_dataset,
                               'test': test_dataset})
    DatasetDict.save_to_disk(hf_datasets, args.datadict_path)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str, default="multi_news")
    parser.add_argument("--qasrl_path", type=str, default="/home/nlp/wolhanr/data/multi_news/data_qasrl")
    parser.add_argument(
        "--dataset_cache_dir", type=str, default="/home/nlp/wolhanr/cache/"
    )
    parser.add_argument("--datadict_path", type=str, default="/home/nlp/wolhanr/data/multi_news/hf_dataset_qasrl")

    args = parser.parse_args()
    main(args)