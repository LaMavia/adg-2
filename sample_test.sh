rates=$(seq 10 10 100)
resample=10

(printf "rate;auc_roc\n";
for rate in $rates; do
  for seed in $(seq $resample); do
      score=$(RATE="$rate" SEED="$seed" python3 \
      ./testerka.py \
      ./classifier-sample_test.py \
      test0_ground_truth.tsv \
      train0_data.tsv \
      test0_data.tsv \
      2>/dev/null \
      | grep 'Average AUC-ROC across all classes: ' \
      | sed 's/Average AUC-ROC across all classes: //')
      printf "%d;%s\n" "$rate" "$score"
  done
done) | tee ./sample_test.csv
