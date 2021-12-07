from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
from datasets import load_dataset
import argparse
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
from tqdm import tqdm
import os
import jsonlines
import json

def write_oie(args, dataset, splitter, column_text, tgt_text, out_dir, predictor, set_name = 'test'):
    if not os.path.isdir(os.path.join(out_dir, set_name, column_text)):
        os.mkdir(os.path.join(out_dir, set_name, column_text))
    if not os.path.isdir(os.path.join(out_dir, set_name, tgt_text)):
        os.mkdir(os.path.join(out_dir, set_name, tgt_text))
    range_ = range(len(dataset[column_text])) if args.run_split == -1 else range(args.run_split*len(dataset[column_text])//10, min(args.run_split*len(dataset[column_text])//10+len(dataset[column_text])//10+1, len(dataset[column_text])-1))
    for i in tqdm(range_):
        if os.path.isfile(os.path.join(out_dir, set_name, tgt_text, f"{i}.jsonl")) and os.path.isfile(os.path.join(out_dir, set_name, column_text, f"{i}.jsonl")):
            continue
        #split the documents
        all_docs = dataset[column_text][i].split("|||||")[:-1]
        sentences_src = splitter.batch_split_sentences(all_docs)
        sentences_summary = splitter.split_sentences(dataset[tgt_text][i])
        f_tgt = jsonlines.open(os.path.join(out_dir, set_name, tgt_text, f"{i}.jsonl"), 'w')
        json_input_tgt = [{"sentence": sentences_summary[k]} for k in range(len(sentences_summary))]
        result_list = predictor.predict_batch_json(json_input_tgt)
        f_tgt.write(result_list)
        f_src = jsonlines.open(os.path.join(out_dir, set_name, column_text, f"{i}.jsonl"), 'w')
        for j, doc in enumerate(all_docs):
            json_input_src = [{"sentence": sentences_src[j][k]} for k in range(len(sentences_src[j]))]
            result_list = predictor.predict_batch_json(json_input_src)
            f_src.write(result_list)
        f_src.close()
        f_tgt.close


def main(args):
    path_oie_model = "https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz"
    predictor = Predictor.from_path(path_oie_model)
    splitter = SpacySentenceSplitter()
    if args.dataset_name in ["multi_news", "multi_x_science_sum"]:
        hf_datasets = load_dataset(args.dataset_name, cache_dir=args.dataset_cache_dir)
        column_text = 'document'
        tgt_text = 'summary'
        out_dir = '/home/nlp/wolhanr/data/multi_news/oie'
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
    # extract oie of each multi doc on train , val and test set

    if args.set_oie == 'all':
        train = hf_datasets['train']
        val = hf_datasets['validation']
        test = hf_datasets['test']
        write_oie(args, test, splitter, column_text, tgt_text, out_dir, predictor, 'test')
        write_oie(args, train, splitter, column_text, tgt_text, out_dir, predictor, 'train')
        write_oie(args, val, splitter, column_text, tgt_text, out_dir, predictor, 'val')
    else:
        dataset = hf_datasets[args.set_oie]
        write_oie(args, dataset, splitter, column_text, tgt_text, out_dir, predictor, args.set_oie)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="multi_news")
    parser.add_argument(
        "--dataset_cache_dir", type=str, default="/home/nlp/wolhanr/cache/"
    )
    parser.add_argument("--set_oie",
                        type=str,
                        default='all',
                        choices=["train", "test", "validation", "all"]
                        )
    parser.add_argument("--run_split",
                        type=int,
                        default=-1,
                        help='divise the dataset on 10 parts and run only the part --run_split',
                        choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                        )
    args = parser.parse_args()

    main(args)