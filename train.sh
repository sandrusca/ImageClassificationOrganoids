python retrain.py \
  --bottleneck_dir=tf_files/bottlenecks \
  --how_many_training_steps=1000 \
  --learning_rate=0.01 \
  --model_dir=inception \
  --summaries_dir=tf_files/training_summaries/basic \
  --output_graph=tf_files/retrained_graph.pb \
  --output_labels=tf_files/retrained_labels.txt \
  --image_dir=training_dataset \
  --validation_percentage=10 \
  --testing_percentage=10 \
  --eval_step_interval=20 \
  --validation_batch_size=-1 \
  --test_batch_size=-1 \
  --print_misclassified_test_images


  # --flip_left_right \
  # --random_crop=10 \
  # --random_scale=5 \
  # --random_brightness=10