#!/usr/bin/python3
import sys
import os
import pandas as pd
from sklearn.metrics import accuracy_score

print('Hello! I am a task number: ', sys.argv[1])
print(f"Current Working Directory: {os.getcwd()}")

data_file_name = "test data"
print(f"Running {data_file_name}")

def analysis():
    ground_truths = [1, 2, 2]
    predicted_labels = [1, 2, 1]
    
    accuracy = accuracy_score(ground_truths, predicted_labels)
    print(f"accuracy : {accuracy}")
    path = os.path.join(os.getcwd(), "myresults.csv")
    results_df = pd.DataFrame({"preds": predicted_labels, "true": ground_truths})
    results_df.to_csv(path, index=False)

    return None
analysis()