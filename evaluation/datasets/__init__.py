"""Dataset adapters: one file per benchmark, responsible for only two things -- how to load and how to score.

The middle part (ingest turn by turn -> search per question -> answer from memory) is identical for all datasets
and lives in run.py, so numbers from different benchmarks are comparable.

Adding a benchmark = add a file in this directory implementing the two functions below, then register it in DATASETS:

    def load(path: str) -> list[Conversation]      # load into the common structure
    def score(q: Question, answer: str, judge) -> Score   # judge whether this question is right

judge is injected by run.py with signature ``judge(prompt: str) -> str`` -- datasets score differently
(some compare with a gold answer, some go through a rubric), but they all use the same judge model.
"""
from dataclasses import dataclass, field


@dataclass
class Turn:
    """One utterance in a conversation."""
    speaker: str
    text: str
    #: When this utterance really happened (ISO, e.g. "2023-05-08"). Required -- memories are ordered by time;
    #: if omitted when backfilling history, the store holds only the evaluation day's timestamp and temporal questions break.
    observed_at: str = ""


@dataclass
class Question:
    id: str
    text: str
    answer: str = ""                              # gold answer (used for scoring when present)
    rubric: list[str] = field(default_factory=list)   # rubric points (for AudioMC-style datasets)
    category: str = ""                            # question type, for per-category scores
    meta: dict = field(default_factory=dict)


@dataclass
class Conversation:
    """One full conversation + the questions about it.

    Each conversation gets its **own memory store** during evaluation (see run.py) -- mixing memories from different
    conversations would secretly feed answers to the model, making the score meaningless.
    """
    id: str
    turns: list[Turn]
    questions: list[Question]


@dataclass
class Score:
    correct: float          # 1/0, or a fraction such as the share of rubric points met
    total: float = 1.0      # full marks for this question (number of points for rubric questions)
    note: str = ""          # scoring rationale, written to the results file for human review


#: benchmark name -> module path. To add one: write a file modelled on locomo.py implementing load() and
#: score(), and register one line here. The CLI's --dataset uses these keys as choices directly,
#: so a misspelled name fails at argument parsing and --help lists them automatically.
DATASETS = {
    "locomo": "evaluation.datasets.locomo",
}


def names() -> list[str]:
    return sorted(DATASETS)


def display_name(name: str) -> str:
    """Display name: NAME from the dataset module, or the CLI key if not set."""
    return getattr(get(name), "NAME", name)


def get(name: str):
    """Get a dataset adapter by name."""
    import importlib
    if name not in DATASETS:
        raise SystemExit(
            f"Unknown dataset: {name}. Available: {', '.join(names())}\n"
            f"To add one: write a file of that name modelled on evaluation/datasets/locomo.py, "
            f"implement load() and score(), then register it in DATASETS here.")
    return importlib.import_module(DATASETS[name])
