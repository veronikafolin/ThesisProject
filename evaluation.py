import argparse
import nltk
import numpy as np
import pandas as pd
from datasets import Dataset, load_metric
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from scipy.stats import entropy
from sklearn.feature_extraction.text import CountVectorizer
import nltk


def get_rouge_metrics(preds, refs):
    """Computes the rouge metrics.

    Args:
        preds: list. The model predictions.
        refs: list. The references.

    Returns:
        The rouge metrics.
    """
    rouge_output = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
    return {
        "r1": round(rouge_output["rouge1"].mid.fmeasure, 4),
        "r2": round(rouge_output["rouge2"].mid.fmeasure, 4),
        "rL": round(rouge_output["rougeL"].mid.fmeasure, 4)
    }


def get_bertscore_metrics(preds, refs):
    """Computes the bertscore metric.

    Args:
        preds: list. The model predictions.
        refs: list. The references.

    Returns:
        The bertscore metrics.
    """

    bertscore_output = bertscore.compute(predictions=preds, references=refs, lang="en")
    return {
        "p": round(np.mean([v for v in bertscore_output["precision"]]), 4),
        "r": round(np.mean([v for v in bertscore_output["recall"]]), 4),
        "f1": round(np.mean([v for v in bertscore_output["f1"]]), 4)
    }


def get_redundancy_scores(preds):
    sum_unigram_ratio = 0
    sum_bigram_ratio = 0
    sum_trigram_ratio = 0
    all_unigram_ratio = []
    all_bigram_ratio = []
    all_trigram_ratio = []

    sum_redundancy = 0
    stop_words = set(stopwords.words("english"))
    count = CountVectorizer()
    all_redundancy = []

    number_file = len(preds)

    for p in preds:
        all_txt = []
        all_txt.extend(word_tokenize(p.strip()))

        # uniq n-gram ratio
        all_unigram = list(ngrams(all_txt, 1))
        uniq_unigram = set(all_unigram)
        unigram_ratio = len(uniq_unigram) / len(all_unigram)
        sum_unigram_ratio += unigram_ratio

        all_bigram = list(ngrams(all_txt, 2))
        uniq_bigram = set(all_bigram)
        bigram_ratio = len(uniq_bigram) / len(all_bigram)
        sum_bigram_ratio += bigram_ratio

        all_trigram = list(ngrams(all_txt, 3))
        uniq_trigram = set(all_trigram)
        trigram_ratio = len(uniq_trigram) / len(all_trigram)
        sum_trigram_ratio += trigram_ratio

        all_unigram_ratio.append(unigram_ratio)
        all_bigram_ratio.append(bigram_ratio)
        all_trigram_ratio.append(trigram_ratio)

        # NID score
        num_word = len(all_txt)
        new_all_txt = [w for w in all_txt if not w in stop_words]
        new_all_txt = [' '.join(new_all_txt)]

        #print(all_txt)
        #print(any(w not in stop_words for w in all_txt))
        try:
            x = count.fit_transform(new_all_txt)
            bow = x.toarray()[0]
            max_possible_entropy = np.log(num_word)
            e = entropy(bow)
            redundancy = (1-e/max_possible_entropy)
            sum_redundancy += redundancy
            all_redundancy.append(redundancy)
        except ValueError:
            continue 

    print(f"Number of documents: {number_file}, average unique unigram ratio is {round(sum_unigram_ratio/number_file, 4)}, average unique bigram ratio is {round(sum_bigram_ratio/number_file, 4)}, average unique trigram ratio is {round(sum_trigram_ratio/number_file, 4)}, NID score is {round(sum_redundancy/number_file, 4)}.")
    return all_unigram_ratio, all_bigram_ratio, all_trigram_ratio, all_redundancy


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--bertscore", default=False, action="store_true", help="If evaluate with bertscore")
    parser.add_argument("--dataset", default="billsum", help="The dataset to use")
    parser.add_argument("--eval_with_bart", default=False, action="store_true", help="If eval with just bart")
    parser.add_argument("--num_epochs", type=int, default=1, help="The number of epochs")
    parser.add_argument("--max_len_source", type=int, default=512, help="The input max size")
    parser.add_argument("--max_len_summary", type=int, default=128, help="The output max size")
    parser.add_argument("--n_docs", type=int, default=0, help="The number of training examples")
    parser.add_argument("--random_chunks", default=False, action="store_true", help="If use random chunks in retrieval")
    parser.add_argument("--redundancy", default=False, action="store_true", help="If evaluate with redundancy")
    parser.add_argument("--rouge", default=False, action="store_true", help="If evaluate with rouge")
    parser.add_argument("--sentence_mode", default=False, action="store_true", help="If use sentences in the retrieval")
    parser.add_argument("--bart_base", default=False, action="store_true", help="If eval with just bart base")
    args = parser.parse_args()

    data_dir = "data/"
    predictions_path = "predictions/" + args.dataset + "_" + str(args.max_len_source) + "_" + \
                       str(args.max_len_summary) + "_" + str(args.num_epochs) + "_epochs"

    if args.bart_base:
        predictions_path += "_bart_base"

    if args.n_docs > 0:
        predictions_path += "_" + str(args.n_docs)

    if args.sentence_mode:
        predictions_path += "_sent"

    if args.random_chunks:
        predictions_path += "_random"

    if args.eval_with_bart:
        predictions_path += "_bart"

    predictions = pd.read_csv(predictions_path)["prediction"].tolist()
    print(len(predictions))
    test_dataset = Dataset.from_pandas(pd.read_csv(data_dir + "billsum_test_set"))

    if args.bertscore:
        bertscore = load_metric("bertscore")
        bertscore_metrics = get_bertscore_metrics(preds=predictions, refs=test_dataset["summary"])
        print(f"\nBERTSCORE-p: {bertscore_metrics['p']}\nBERTSCORE-r: {bertscore_metrics['r']}\nBERTSCORE-f1: {bertscore_metrics['f1']}")

    if args.redundancy:
        nltk.download("stopwords")
        nltk.download("punkt")
        get_redundancy_scores(predictions)

    if args.rouge:
        rouge = load_metric("rouge")
        rouge_metrics = get_rouge_metrics(preds=predictions, refs=test_dataset["summary"])
        print(f"\nROUGE-1: {rouge_metrics['r1']}\nROUGE-2: {rouge_metrics['r2']}\nROUGE-L: {rouge_metrics['rL']}")


