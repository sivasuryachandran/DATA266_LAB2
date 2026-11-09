"""python -m p1_rag.run_competition  -> submission.csv (reads QUESTIONS_PATH, default Questions.csv)."""
import os

from common.run_info import print_run_info
from p1_rag.ingest import find_data_dir
from p1_rag.pipeline import answer_file, build

if __name__ == "__main__":
    print_run_info()
    data_dir = find_data_dir()
    qpath = os.environ.get("QUESTIONS_PATH", os.path.join(data_dir, "Questions.csv"))
    system, _, _ = build(data_dir)
    answer_file(system, qpath, "submission.csv")
