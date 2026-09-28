"""Faithful port of the Just-Read-Twice (Arora et al. 2024) recall protocol that GDN-2 Table 4
follows ("input length truncated to 2K"): external/prefix-linear-attention/lm-eval-harness,
launch_jrt.py --context_length 2048 --answer_length 50 --cutting_context.

  * cutting_context (api/task.py:truncate_context): tokenize doc[context_key]; locate the answer
    (earliest match over answer_key fields; TQA/DROP have none -> position 0 = document HEAD; NQ
    falls back to the dataset's tok_pos); keep a (2048-50)=1998-token window CENTERED on it
    (half before, rest after); decode back to text.
  * doc_to_text: "<context>. <question.strip()>" (no trailing space; FDA/SWDE: "<text>. Key:";
    SWDE drops the "Summary of information above..." block; TQA strips [PAR]/[DOC]/[TLE]).
  * generate_until: until ["\\n"], max_gen_toks 48; metric = case-insensitive contains.
  * datasets: based-swde-v2, based-fda, based-squad, based_triviaqa, based_nq_2048, based_drop.
Tokenizer for the cut = the model's (Llama-2 / TinyLlama for GDN-2 == data/llama2_tok here)."""
import os
import re
from functools import lru_cache

import numpy as np
from lm_eval.api.instance import Instance
from lm_eval.api.task import ConfigurableTask

CONTEXT_LENGTH = int(os.environ.get("JRT_CONTEXT_LENGTH", 2048))
ANSWER_LENGTH = int(os.environ.get("JRT_ANSWER_LENGTH", 50))
# JRT_NO_STRIP=1 preserves the source's LEADING whitespace (the attention-sink prefix our
# model depends on — see retrieval_protocol_investigation.md §4). Answer-centered cut and
# SWDE summary-block removal are KEPT; only the leading .strip() is disabled.
NO_STRIP = os.environ.get("JRT_NO_STRIP", "0") == "1"
# JRT_PREPEND_SPACES=N: strip as usual, then prepend N literal spaces (sweep the
# leading-whitespace attention-sink effect — find the threshold N that recovers SQuAD).
PREPEND_SPACES = int(os.environ.get("JRT_PREPEND_SPACES", "0"))


def _lead(text):
    """The leading-whitespace prefix to re-attach under NO_STRIP (else '')."""
    return text[:len(text) - len(text.lstrip())] if NO_STRIP else ""
TOK_DIR = os.environ.get("JRT_TOK_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))), "data", "llama2_tok"))


@lru_cache(maxsize=1)
def _tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOK_DIR)


def contains_score(prediction, labels):
    return max(int(bool(re.search(re.compile(re.escape(label), re.IGNORECASE), prediction)))
               for label in labels)


def cut_context(doc, context_key, answer_key):
    """Port of truncate_context (JRT lm-eval-harness api/task.py). Returns a COPY of doc with
    doc[context_key] replaced by the answer-centered window (or the doc unchanged if short)."""
    tok = _tok()
    desired = CONTEXT_LENGTH - ANSWER_LENGTH
    context = doc[context_key] if NO_STRIP else doc[context_key].strip()
    doc_tokens = tok(context, add_special_tokens=True)["input_ids"]        # batch_encode_plus default
    answer_pos = -1
    for key in answer_key:
        answer = doc[key]
        if isinstance(answer, list):
            answer = answer[0]
        if answer == "" or len(answer) <= 1:
            return None                                                      # JRT drops the doc
        m = re.compile(re.escape(answer), re.IGNORECASE).search(context)
        if m and (answer_pos == -1 or answer_pos > m.start()):
            answer_pos = m.start()
    if answer_pos < 0 and "alt_answers" in doc:
        for answer in doc["alt_answers"]:
            m = re.compile(re.escape(answer), re.IGNORECASE).search(context)
            if m and (answer_pos == -1 or answer_pos > m.start()):
                answer_pos = m.start()
            if answer_pos >= 0:
                break
    if len(answer_key) == 0:
        answer_pos = 0
    context_short_tokens = tok(context[:answer_pos], add_special_tokens=True)["input_ids"]
    answer_tok_pos = len(context_short_tokens)
    if answer_pos == -1 and "tok_pos" in doc and len(context_short_tokens) > desired:
        answer_tok_pos = int(doc["tok_pos"])
    half = desired // 2
    start = max(0, answer_tok_pos - half)
    remaining = desired - (answer_tok_pos - start)
    end = min(len(doc_tokens), answer_tok_pos + remaining)
    short = tok.decode(doc_tokens[start:end], skip_special_tokens=True)
    out = dict(doc)
    out[context_key] = short
    return out


class _JRT(ConfigurableTask):
    VERSION = 0
    DATASET_NAME = "default"
    CONTEXT_KEY = "context"
    ANSWER_KEY = []

    def __init__(self, **kwargs):
        super().__init__(config={"metadata": {"version": self.VERSION}})

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def validation_docs(self):
        ds = self.dataset["validation"]
        # JRT_SHARD=i/n: run doc i, i+n, i+2n, ... (parallelize one task over n GPUs;
        # merge the shards' TEXT dumps with analysis/merge_shards.py — every shard
        # is the same size +-1 so the mean of shard means ~= the pooled mean).
        sh = os.environ.get("JRT_SHARD")
        if sh:
            i, n = (int(x) for x in sh.split("/"))
            ds = ds.select(range(i, len(ds), n))
        return ds

    def _cut(self, doc):
        c = cut_context(doc, self.CONTEXT_KEY, self.ANSWER_KEY)
        return doc if c is None else c

    def construct_requests(self, doc, ctx, chat_template=None, apply_chat_template=False, **kwargs):
        return [Instance(request_type="generate_until", doc=doc,
                         arguments=(ctx, {"until": ["\n"], "max_gen_toks": 48}), idx=0, **kwargs)]

    def _labels(self, doc):
        return list(doc["answers"]) if "answers" in doc else [doc["value"]]

    def process_results(self, doc, results):
        return {"contains": contains_score(results[0], self._labels(doc))}

    def aggregation(self):
        return {"contains": np.mean}

    def higher_is_better(self):
        return {"contains": True}


class JRTSquad(_JRT):
    DATASET_PATH = "hazyresearch/based-squad"
    CONTEXT_KEY = "text"
    ANSWER_KEY = ["value"]

    def doc_to_text(self, doc):
        t = self._cut(doc)["text"]
        t = t if NO_STRIP else t.strip()
        return " " * PREPEND_SPACES + t

    def doc_to_target(self, doc):
        return doc["value"]


class JRTFDA(_JRT):
    DATASET_PATH = "hazyresearch/based-fda"
    CONTEXT_KEY = "text"
    ANSWER_KEY = ["key", "value"]

    def doc_to_text(self, doc):
        doc = self._cut(doc)
        text = doc["text"]
        lead = _lead(text)
        question = doc["key"] + ":"
        while text.lower().endswith(question.lower()):
            text = text[:-len(question)]
        question = doc["key"][0].upper() + doc["key"][1:] + ":"
        text = text.strip("\n").strip(".")
        out = text
        if not out.endswith("."):
            out += "."
        return lead + out + " " + question

    def doc_to_target(self, doc):
        return doc["value"]


class JRTSWDE(_JRT):
    DATASET_PATH = "hazyresearch/based-swde-v2"
    CONTEXT_KEY = "text"
    ANSWER_KEY = ["key", "value"]

    def doc_to_text(self, doc):
        doc = self._cut(doc)
        text = doc["text"]
        lead = _lead(text)
        while text.find("Summary of information above...") >= 0:
            text = text.split("Summary of information above...")[0]
        question = doc["key"] + ":"
        while text.lower().endswith(question.lower()):
            text = text[:-len(question)]
        question = doc["key"][0].upper() + doc["key"][1:] + ":"
        out = text.strip("\n").strip().strip("\n").strip().strip(".") + "."
        return lead + out + " " + question

    def doc_to_target(self, doc):
        return doc["value"]


class _JRTQA(_JRT):
    CONTEXT_KEY = "context"

    def _clean(self, context):
        return context

    def doc_to_text(self, doc):
        doc = self._cut(doc)
        lead = _lead(doc["context"])
        context = self._clean(doc["context"].strip())
        question = doc["question"].strip()
        while context.lower().endswith(question.lower()):
            context = context[:-len(question)]
        return lead + context.strip().strip(".") + ". " + question

    def doc_to_target(self, doc):
        return list(doc["answers"])[0]


class JRTTriviaQA(_JRTQA):
    DATASET_PATH = "hazyresearch/based_triviaqa"
    ANSWER_KEY = []                       # JRT: no answer key -> window from the document HEAD

    def _clean(self, context):
        return re.sub(r"\[PAR\]|\[DOC\]|\[TLE\]", "", context)


class JRTNQ(_JRTQA):
    DATASET_PATH = "hazyresearch/based_nq_2048"
    ANSWER_KEY = ["answers"]


class JRTDrop(_JRTQA):
    DATASET_PATH = "hazyresearch/based_drop"
    ANSWER_KEY = []
