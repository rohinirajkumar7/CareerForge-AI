from rag.bm25 import BM25, tokenize


def test_tokenizer_keeps_tech_tokens_and_splits_separators():
    assert tokenize("C++ and C#, Node.js") == ["c++", "c#", "node.js"]
    assert tokenize("CI/CD pipelines") == ["ci", "cd", "pipeline"]
    assert tokenize("Full-Stack Developer") == ["full", "stack", "developer"]


def test_tokenizer_drops_stopwords_and_strips_plurals():
    assert tokenize("the skills of APIs") == ["skill", "api"]
    assert tokenize("class") == ["class"]


def test_rarer_terms_weigh_more():
    bm = BM25([["python", "sql"], ["python", "docker"], ["python", "terraform"]])
    assert bm.idf("terraform") > bm.idf("python")


def test_ranks_matching_document_first():
    docs = [tokenize(t) for t in ["react node express", "terraform aws vpc", "pandas sql tableau"]]
    bm = BM25(docs)
    assert bm.top_k(tokenize("terraform vpc"), 3)[0][0] == 1


def test_term_frequency_saturates_and_length_is_normalized():
    short = ["docker", "kubernetes"]
    long = ["docker"] + ["filler"] * 50
    bm = BM25([short, long])
    scores = bm.scores(["docker"])
    assert scores[0] > scores[1]


def test_unknown_terms_and_empty_query_score_zero():
    bm = BM25([["python"], ["java"]])
    assert bm.top_k(["haskell"], 3) == []
    assert bm.top_k([], 3) == []


def test_ties_break_by_index():
    bm = BM25([["python"], ["python"]])
    assert [i for i, _ in bm.top_k(["python"], 2)] == [0, 1]


def test_empty_corpus_does_not_crash():
    assert BM25([]).top_k(["x"], 3) == []
