import os
import sys

import pandas as pd
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Disable tensorflow compilation warnings
import tensorflow.compat.v1 as tf

# adapted from original code
# here I can read the directory and do a for loop for every image in a folder
directory = sys.argv[1]

# array with file names
file_names = []
# empty array for testing labels
y_true = []
# empty array for predicted labels
y_pred = []

# binary
labels = ['good', 'bad']
y_scores_good = []
y_scores_bad = []

# multi
'''
labels = ["good1", "bad1", "good2", "bad2", "good3", "bad3"]
# empty array for the scores per label
y_scores1_good = []
y_scores1_bad = []
y_scores2_good = []
y_scores2_bad = []
y_scores3_good = []
y_scores3_bad = []
'''
# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line
               in tf.gfile.GFile("tf_files/retrained_labels.txt")]

# Unpersists graph from file
with tf.gfile.GFile("tf_files/retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

# adapted from original code
for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        image_path = os.path.join(directory, filename)
        print(image_path)
        image_data = tf.gfile.GFile(image_path, 'rb').read()
        current_file = filename.split(".")[0]
        file_names.append(current_file)
        current_label = current_file.split("_")[1]
        print(current_label)
        y_true.append(current_label)
        with tf.Session() as sess:
            # Feed the image_data as input to the graph and get first prediction
            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

            predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})

            # Sort to show labels of first prediction in order of confidence
            top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

            # save first top_k in an array with predicted labels here
            y_pred.append(label_lines[top_k[0]])

            # print all predicted results in command line
            # save all predicted results in lists
            for node_id in top_k:
                human_string = label_lines[node_id]
                score = predictions[0][node_id]
                # binary
                if "good" == human_string:
                    y_scores_good.append(score)
                if "bad" == human_string:
                    y_scores_bad.append(score)
                # multi
                '''
                if "good1" == human_string:
                    y_scores1_good.append(score)
                if "bad1" == human_string:
                    y_scores1_bad.append(score)
                if "good2" == human_string:
                    y_scores2_good.append(score)
                if "bad2" == human_string:
                    y_scores2_bad.append(score)
                if "good3" == human_string:
                    y_scores3_good.append(score)
                if "bad3" == human_string:
                    y_scores3_bad.append(score)
                '''
                print('%s (score = %.5f)' % (human_string, score))

        current_label = ""
        continue
    else:
        continue


# binary
data = pd.DataFrame(np.column_stack([file_names, y_true, y_pred, y_scores_good, y_scores_bad]),
                    columns=['file_names', 'y_true', 'y_pred', 'y_scores_good', 'y_scores_bad'])

data.to_csv('data_stage3.csv', index=False, header=['file_names', 'y_true', 'y_pred', 'y_scores_good', 'y_scores_bad'])

# multi
'''
data = pd.DataFrame(np.column_stack([file_names, y_true, y_pred, y_scores1_good, y_scores1_bad, y_scores2_good, y_scores2_bad,
                                     y_scores3_good, y_scores3_bad]),
                    columns=['file_names', 'y_true', 'y_pred', 'y_scores1_good', 'y_scores1_bad', 'y_scores2_good', 'y_scores2_bad',
                             'y_scores3_good', 'y_scores3_bad'])

data.to_csv('data_multi.csv', index=False, header=['file_names', 'y_true', 'y_pred', 'y_scores1_good', 'y_scores1_bad',
                                                   'y_scores2_good', 'y_scores2_bad', 'y_scores3_good',
                                                   'y_scores3_bad'])
'''