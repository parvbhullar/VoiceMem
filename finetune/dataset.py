import json

from utils import system_prompt

MAX_HISTORY = 6
LANGS = ("en",)
CATEGORIES = ("knowledge", "emotion", "persona", "casual")


class DialogueDataset:
    def __init__(self, path):
        self.path = path

        with open(path) as f:
            self.rows = [json.loads(line) for line in f if line.strip()]

        for i, row in enumerate(self.rows):
            check(row, i)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return prompt(row), answer(row)


def check(row, i=0):
    msgs = row["messages"]

    assert msgs[0]["role"] == "system", f"row {i}: first message is not system"
    assert msgs[-1]["role"] == "assistant", f"row {i}: last message is not assistant"
    assert len(msgs) % 2 == 1, f"row {i}: messages after system are not user/assistant pairs"

    for j, m in enumerate(msgs[1:]):
        want = "user" if j % 2 == 0 else "assistant"
        assert m["role"] == want, f"row {i}, message {j + 1} should be {want}"
        assert isinstance(m["content"], str), f"row {i}, message {j + 1}: content is not a string"

    assert history_turns(row) <= MAX_HISTORY, f"row {i}: history exceeds {MAX_HISTORY} turns"

    meta = row.get("meta", {})
    assert meta.get("lang") in LANGS, f"row {i}: invalid lang: {meta.get('lang')}"
    assert meta.get("category") in CATEGORIES, f"row {i}: invalid category: {meta.get('category')}"

    want = system_prompt(meta["category"], meta["lang"])
    assert msgs[0]["content"] == want, f"row {i}: system prompt does not match category/lang"


def history_turns(row):
    return (len(row["messages"]) - 3) // 2


def question(row):
    return row["messages"][-2]["content"]


def answer(row):
    return row["messages"][-1]["content"]


def prompt(row):
    return row["messages"][:-1]


def load(path):
    return DialogueDataset(path).rows


if __name__ == "__main__":
    dataset = DialogueDataset("data/sample.jsonl")

    print("Dataset size:", len(dataset))

    msgs, ref = dataset[0]

    print("Messages:", len(msgs))
    print("Question:", msgs[-1]["content"][:40], "...")
    print("Reference answer:", ref)
