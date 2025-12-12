import sys
import time
import subprocess
from resource import RUSAGE_CHILDREN, getrusage

def main(): 
    if len(sys.argv) != 5:
        print("usage: python3 testerka.py classifier.py ground_truth training test ")
        sys.exit(1)

    _, classifier, ground_truth, training_path, test_path = sys.argv
    output_file = "_output.tsv"

    # run the mapper 
    start_time = time.time()
    subprocess.run(["python3", classifier, training_path, test_path, output_file], check=True, stderr=sys.stderr)
    end_time = time.time() - start_time 

    usage = getrusage(RUSAGE_CHILDREN)
    max_memory_usage = usage.ru_maxrss/1000

    subprocess.run(["python3", "./evaluate.py", output_file, ground_truth], check=True, stderr=sys.stderr, stdout=sys.stdout)


    print(f"\nTotal time: {end_time:.2f} s")
    print(f"Peak memory usage: {max_memory_usage:.3f} MB")


if __name__ == "__main__":
    main()
