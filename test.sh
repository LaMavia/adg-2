#!/usr/bin/env bash

# ./test.sh ground_truth training test 

cd $(dirname $0)

OUTPUT=./_output.tsv
GROUND_TRUTH=$1

shift

command time -f 'command: %C\nmemory: %M KB\ntime: %U s' python ./classifier.py $@ "$OUTPUT" \
  && python3 ./evaluate.py "$OUTPUT" "$GROUND_TRUTH"
