#!/usr/bin/env python

import sys, os.path
import numpy as np
import pandas
import eval_metrics_DF as em
from glob import glob


# submit_file = sys.argv[1]
# truth_dir = sys.argv[2]
# phase = sys.argv[3]

label_dir = sys.argv[1]
score_tag = sys.argv[2]

def eval_to_score_file(score_file, cm_key_file):
    cm_data = pandas.read_csv(cm_key_file, sep=' ', header=None)
    submission_scores = pandas.read_csv(score_file, sep=' ', header=None, skipinitialspace=True)
    #print(len(submission_scores))
    #print(len(cm_data))
    if len(submission_scores) != len(cm_data):
        print('CHECK: submission has %d of %d expected trials.' % (len(submission_scores), len(cm_data)))
        exit(1)

    if len(submission_scores.columns) > 2:
        print('CHECK: submission has more columns (%d) than expected (2). Check for leading/ending blank spaces.' % len(
            submission_scores.columns))
        exit(1)

    cm_scores = submission_scores.merge(cm_data, left_on=0, right_on=0,
                                        how='inner')  # check here for progress vs eval set
    #print(cm_scores.head())
    bona_cm = cm_scores[cm_scores['1_y'] == 'real']['1_x'].values
    spoof_cm = cm_scores[cm_scores['1_y'] == 'fake']['1_x'].values
    eer_cm = em.compute_eer(bona_cm, spoof_cm)[0]
    out_data = "eer: %.2f\n" % (100 * eer_cm)
    print(out_data)
    return eer_cm


if __name__ == "__main__":

    # if not os.path.isfile(submit_file):
    #     print("%s doesn't exist" % (submit_file))
    #     exit(1)

    # if not os.path.isdir(truth_dir):
    #     print("%s doesn't exist" % (truth_dir))
    #     exit(1)

    print('label_dir: ', label_dir)
    print(f'scores_dir: ./scores/scores_C1-C7_{score_tag}.txt')

    print("C1")
    C1_eer = eval_to_score_file(os.path.join(f'./scores/scores_C1_{score_tag}.txt'), os.path.join(label_dir + 'C1.txt'))
    
    print("C2")
    C2_eer = eval_to_score_file(os.path.join(f'./scores/scores_C2_{score_tag}.txt'), os.path.join(label_dir + 'C2.txt'))

    print("C3")
    C3_eer = eval_to_score_file(os.path.join(f'./scores/scores_C3_{score_tag}.txt'), os.path.join(label_dir + 'C3.txt'))

    print("C4")
    C4_eer = eval_to_score_file(os.path.join(f'./scores/scores_C4_{score_tag}.txt'), os.path.join(label_dir + 'C4.txt'))
    
    print("C5")
    C5_eer = eval_to_score_file(os.path.join(f'./scores/scores_C5_{score_tag}.txt'), os.path.join(label_dir + 'C5.txt'))

    print("C6")
    C6_eer = eval_to_score_file(os.path.join(f'./scores/scores_C6_{score_tag}.txt'), os.path.join(label_dir + 'C6.txt'))

    print("C7")
    C7_eer = eval_to_score_file(os.path.join(f'./scores/scores_C7_{score_tag}.txt'), os.path.join(label_dir + 'C7.txt'))

    # print("A1")
    # _ = eval_to_score_file(os.path.join(f'./scores/scores_A1_{score_tag}.txt'), os.path.join(label_dir + 'A1.txt'))
    #
    # print("A2")
    # _ = eval_to_score_file(os.path.join(f'./scores/scores_A2_{score_tag}.txt'), os.path.join(label_dir + 'A2.txt'))

    #print(C1_eer,C2_eer,C3_eer,C4_eer,C5_eer,C6_eer,C7_eer)

    values = [C1_eer, C2_eer, C3_eer, C4_eer, C5_eer, C6_eer, C7_eer]

    print("C1\tC2\tC3\tC4\tC5\tC6\tC7\tavg")
    avg =sum(values)/len(values)
    print("\t".join(f"{v * 100:.2f}" for v in values)+ f"\t{avg * 100:.2f}")
