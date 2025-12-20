import argparse
import sys
import pandas as pd
from sklearn.metrics import roc_auc_score
import multiprocessing as mp
import re

def calculate_auc_roc(output_file, ground_truth_file):
    output = pd.read_csv(output_file, sep='\t')
    ground_truth = pd.read_csv(ground_truth_file, sep='\t')

    # Ensure that FASTA files match
    if not all(output.iloc[:, 0] == ground_truth.iloc[:, 0]):
        raise ValueError("FASTA files in output and ground truth files do not match.")

    # Get the list of classes from the output file (excluding the column with FASTA file names)
    classes = output.columns[1:]

    # Calculate AUC-ROC for each class
    auc_scores = []
    for this_class in classes:
        # Ground truth binary labels for the current class
        true_labels = (ground_truth.iloc[:, 1] == this_class).astype(int)

        # Predicted values for the current class
        predicted_scores = output[this_class]

        # Calculate AUC-ROC if there is at least one positive and one negative label
        if true_labels.nunique() > 1:
            auc = roc_auc_score(true_labels, predicted_scores)
            auc_scores.append(auc)
            # print(f"AUC-ROC for class {this_class}: {auc:.4f}")
        else:
            pass
            # print(f"Skipping class {this_class} due to lack of positive/negative samples.")

    # Compute the average AUC-ROC
    average_auc = sum(auc_scores) / len(auc_scores) if auc_scores else 0.0
    # print(f"Average AUC-ROC across all classes: {average_auc:.4f}")
    return average_auc

def parse_path(path: str) -> tuple[bool, int, int] | None:
    regex = r"(test_|)output-(\d{3})-(\d{2}).tsv"
    for m in re.finditer(regex, path):
        return m.group(1) == "test_", int(m.group(2)), int(m.group(3))

    return None


def main():
    _, gt, *outputs = sys.argv 

    outputs = [(o, p) for o in outputs if (p := parse_path(o)) is not None]

    # parser = argparse.ArgumentParser(description="Calculate average AUC-ROC from classifier output and ground truth.")
    # parser.add_argument("classifier_output", type=str, help="TSV file containing classifier output.")
    # parser.add_argument("testing_ground_truth", type=str, help="TSV file containing ground truth classification.")
    # args = parser.parse_args()

    p = mp.Pool(10)
    with p:
        scores = p.starmap(calculate_auc_roc, [(out, gt) for out, *_ in outputs])
        
    test_data = {
        "rate": [],
        "auc_roc": []
    } 

    train_data = {
        "rate": [],
        "auc_roc": []
    } 

    for score, (_, (is_test, rate, _)) in zip(scores, outputs):
        if is_test:
            d = test_data
        else:
            d = train_data

        d['rate'].append(rate)
        d['auc_roc'].append(score)


    pd.DataFrame(test_data).to_csv("./1-sample-test.tsv", sep="\t")
    pd.DataFrame(train_data).to_csv("./1-sample-train.tsv", sep="\t")

if __name__ == "__main__":
    main()
