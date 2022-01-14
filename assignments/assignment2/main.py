import time
from assignments.assignment2.a_classification import run_classification
from assignments.assignment2.b_regression import run_regression
from assignments.assignment2.c_clustering import run_clustering
from assignments.assignment2.d_custom import run_custom

if __name__ == "__main__":
    start = time.time()
    print("Being....\n")
    run_classification()
    run_regression()
    run_clustering()
    run_custom()

    end = time.time()
    run_time = round(end - start)
    print("End....\n")
    print(f"{30 * '#'}\nTotal run_time:{run_time}s\n{30 * '#'}\n")


# Cheers :)
