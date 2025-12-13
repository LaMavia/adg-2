rates=$(seq 10 10 100)
resample=10


(printf "rate;auc_roc\n";
for rate in $rates; do
  for _ in $(seq $resample); do
      score=$(RATE="$rate" python3 \
      ./testerka.py \
      ./classifier-sample_train.py \
      test0_ground_truth.tsv \
      train0_data.tsv \
      test0_data.tsv \
      2>/dev/null \
      | grep 'Average AUC-ROC across all classes: ' | sed 's/Average AUC-ROC across all classes: //')
      printf "%d;%s\n" "$rate" "$score"
  done
done) | tee ./sample_train.csv
