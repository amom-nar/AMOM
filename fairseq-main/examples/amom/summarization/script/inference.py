from rouge_score import rouge_scorer
import numpy as np


def rouge(hypothesis, references):
    """
    calculate rouge metric: rouge-1, rouge-2, rouge-L
    :param hypothesis: list of str
    :param references: list of str
    :return:
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    total_score = []
    for hyp, ref in zip(hypothesis, references):
        res = scorer.score(hyp, ref)
        total_score.append([res['rouge1'], res['rouge2'], res['rougeL']])
    total_score = np.mean(total_score, axis=0)
    return tuple(total_score)


def remove_repeat_n_gram(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as fin:
        fout = open('{}'.format(output_file), 'w', encoding='utf-8')
        for line in fin:
            line = line.strip()
            words = line.split(' ')
            words_filtered = []
            for w in words:
                if len(words_filtered) >= 1 and w == words_filtered[-1]:
                    continue
                else:
                    words_filtered.append(w)
            # line = ' '.join(words_filtered).replace(' ##', '')
            line = ' '.join(words_filtered)
            words = line.split(' ')
            words_filtered = []
            for w in words:
                if len(words_filtered) >= 1 and w == words_filtered[-1]:
                    continue
                else:
                    words_filtered.append(w)
            fout.write('{}\n'.format(' '.join(words_filtered)))


if __name__ == '__main__':
    # inputfile: generate results
    # output_file: results removing the repeated n-gram tokens 
    # test_target: file containing test.target data
    remove_repeat_n_gram(inputfile, output_file)

    gold = []
    decode_result = []
    with open(test_target, encoding="utf-8") as f, open("output_file", encoding="utf-8") as hf:
        gold_data = f.readlines()
        for data in gold_data:
            gold.append(data)
        hypo_data = hf.readlines()
        for data in hypo_data:
            decode_result.append(data)
    assert len(gold) == len(decode_result)
    print("rouge: {}".format(rouge(gold, decode_result)))

    # files2rouge.run("test.target.tokenized", "xsum_result_1.hypo.tokenized")