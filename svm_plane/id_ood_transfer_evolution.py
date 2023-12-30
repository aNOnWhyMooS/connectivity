import sys
import pickle
import numpy as np
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_recall_fscore_support

from datasets import load_dataset, concatenate_datasets
from transformers import BertTokenizer

seed = sys.argv[1]
steps = sys.argv[2]
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
paws_data = load_dataset(
    "csv", data_files={"dev_and_test": "../paws_final/dev_and_test.tsv"}, delimiter="\t"
)
paws_data_to_labels = {}
for sample in paws_data["dev_and_test"]:
    ids = tokenizer(
        sample["sentence1"],
        sample["sentence2"],
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )["input_ids"][0]
    paws_data_to_labels[tokenizer.decode(ids[ids != tokenizer.pad_token_id])] = sample[
        "label"
    ]
paws_data = paws_data_to_labels

qqp_data = load_dataset("glue", "qqp")["validation"]
qqp_data_to_labels = {}
datasets = []
for i in range(len(qqp_data.features["label"].names)):
    datasets.append(
        qqp_data.shuffle(seed=42)
        .filter(lambda e: e["label"] == i)
        .select(list(range(512)))
    )
qqp_data = concatenate_datasets(datasets)
qqp_data_to_labels = {}
for sample in qqp_data:
    ids = tokenizer(
        sample["question1"],
        sample["question2"],
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )["input_ids"][0]
    qqp_data_to_labels[tokenizer.decode(ids[ids != tokenizer.pad_token_id])] = sample[
        "label"
    ]
qqp_data = qqp_data_to_labels


def load_data(pkl_file, dataset="paws"):
    with open(pkl_file, "rb") as f:
        reprs = pickle.load(f)[dataset]
    Xs, Ys = [], []
    print("Check shape of representation:", list(reprs.values())[0].shape)
    print(f"Considering {len(reprs)} samples")
    for k, repr in reprs.items():
        Xs.append(repr.numpy())
        Ys.append(paws_data[k] if dataset == "paws" else qqp_data[k])
    return Xs, Ys


def get_f1(train_Xs, train_Ys, test_Xs, test_Ys):
    clf = make_pipeline(StandardScaler(), svm.SVC(gamma="auto", kernel="linear"))
    clf.fit(np.array(train_Xs), np.array(train_Ys))
    Ypred = clf.predict(test_Xs)
    return f1_score(test_Ys, Ypred), precision_recall_fscore_support(test_Ys, Ypred)


pkl_file = f"../extract_repr/reduced/qqp_paws_embeds_{seed}_{steps}.pkl"
f1, other_scores = get_f1(*load_data(pkl_file, "qqp"), *load_data(pkl_file, "paws"))

print(f"F1 for {seed} at {steps}: {f1}")

print("Precisions:", other_scores[0])
print("Recalls:", other_scores[1])
print("F1s:", other_scores[2])
print("Supports:", other_scores[3])
