import os
import config
import logging
import csv
def get_entities(seq):
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]
    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ['O']):
      if (chunk == 'S'):
        continue
      tag = chunk[0]
      type_ = chunk.split('-')[-1]

      if end_of_chunk(prev_tag, tag, prev_type, type_):
          chunks.append((prev_type, begin_offset, i - 1))
      if start_of_chunk(prev_tag, tag, prev_type, type_):
          begin_offset = i
      prev_tag = tag
      prev_type = type_
    return chunks
def end_of_chunk(prev_tag, tag, prev_type, type_):
    chunk_end = False

    if prev_tag == 'S':
        chunk_end = True
    if prev_tag == 'B' and tag == 'B':
        chunk_end = True
    if prev_tag == 'B' and tag == 'S':
        chunk_end = True
    if prev_tag == 'B' and tag == 'O':
        chunk_end = True
    if prev_tag == 'I' and tag == 'B':
        chunk_end = True
    if prev_tag == 'I' and tag == 'S':
        chunk_end = True
    if prev_tag == 'I' and tag == 'O':
        chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True
    return chunk_end
def start_of_chunk(prev_tag, tag, prev_type, type_):
    chunk_start = False

    if tag == 'B':
        chunk_start = True
    if tag == 'S':
        chunk_start = True

    if prev_tag == 'S' and tag == 'I':
        chunk_start = True
    if prev_tag == 'O' and tag == 'I':
        chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    return chunk_start


def f1_score(y_true, y_pred, mode='dev'):
    true_entities = set(get_entities(y_true))
    pred_entities = set(get_entities(y_pred))
    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)
    nb_true = len(true_entities)

    p = nb_correct / nb_pred if nb_pred > 0 else 0
    r = nb_correct / nb_true if nb_true > 0 else 0
    score = 2 * p * r / (p + r) if p + r > 0 else 0
    if mode == 'dev':
        return score
    else:
        f_score = {}
        for label in config.labels:
            true_entities_label = set()
            pred_entities_label = set()
            for t in true_entities:
                if t[0] == label:
                    true_entities_label.add(t)
            for pr in pred_entities:
                if pr[0] == label:
                    pred_entities_label.add(pr)
            nb_correct_label = len(true_entities_label & pred_entities_label)
            nb_pred_label = len(pred_entities_label)
            nb_true_label = len(true_entities_label)
            p_label = nb_correct_label / nb_pred_label if nb_pred_label > 0 else 0
            r_label = nb_correct_label / nb_true_label if nb_true_label > 0 else 0
            score_label = 2 * p_label * r_label / \
                (p_label + r_label) if p_label + r_label > 0 else 0
            f_score[label] = score_label
        return f_score, score, p, r
def bad_case(y_true, y_pred, data):

    if not os.path.exists(config.case_dir):
        with open(config.case_dir, 'w', encoding='utf-8') as f:
            f.close()
    output = open(config.case_dir, 'w', encoding='utf-8')
    f_score, score, p, r = f1_score(y_true, y_pred, mode='test')
    output.write('Precision:\t\t' + str(p) + '\n')
    output.write('Recall:\t\t' + str(r) + '\n')
    output.write('F1_Score:\t\t' + str(score) + '\n')
    for label in config.labels:
        output.write(f'{label}\t\t{str(f_score[label])}\n')
    for idx, (t, p) in enumerate(zip(y_true, y_pred)):
        if t == p:
            output.write("good case " + str(idx) + ": \n")
            output.write("sentence: " + str(data[idx]) + "\n")
            output.write("golden label: " + str(t) + "\n")
            output.write("model pred: " + str(p) + "\n")
            continue
        else:
            output.write("bad case " + str(idx) + ": \n")
            output.write("sentence: " + str(data[idx]) + "\n")
            output.write("golden label: " + str(t) + "\n")
            output.write("model pred: " + str(p) + "\n")
    logging.info("--------Bad Cases reserved !--------")

def new_bad_case(y_true, y_pred, data):
    if not os.path.exists(config.case_log_dir):
        with open(config.case_log_dir, 'w', encoding='utf-8') as f:
            writer =csv.writer(f)
            writer.writerow(config.labels)
            f.close()
    output = open(config.case_log_dir, 'a+', encoding='utf-8')
    writer =csv.writer(output)
    f_score, score, p, r = f1_score(y_true, y_pred, mode='test')
    entities_f1 = []
    for label in config.labels:
        entities_f1.append(f_score[label])
    entities_f1.append(score)
    writer.writerow(entities_f1)
    logging.info("--------Records reserved !--------")
    bad_case(y_true, y_pred, data)
if __name__ == "__main__":
    y_t = [['O', 'O', 'O', 'B-address', 'I-address',
            'I-address', 'O'], ['B-name', 'I-name', 'O']]
    y_p = [['O', 'O', 'B-address', 'I-address', 'I-address',
            'I-address', 'O'], ['B-name', 'I-name', 'O']]
    sent = [['十', '一', '月', '中', '山', '路', '电'], ['周', '静', '说']]
    bad_case(y_t, y_p, sent)
