import pytest

from rag.knowledge_base import KINDS, Chunk, KnowledgeBase, cited_sources, format_context, get_knowledge_base

APP_ROLES = [
    "AI / Machine Learning Engineer",
    "Data Scientist",
    "Data Analyst",
    "Software Engineer",
    "Full Stack Developer",
    "Cybersecurity Engineer",
    "Cloud Engineer",
    "DevOps Engineer",
]


@pytest.fixture(scope="module")
def kb():
    return get_knowledge_base()


def test_corpus_is_complete_and_ids_unique(kb):
    assert len(kb.chunks) == len(APP_ROLES) * len(KINDS)
    assert {c.role for c in kb.chunks} == set(APP_ROLES)
    for role in APP_ROLES:
        assert {c.kind for c in kb.chunks if c.role == role} == set(KINDS)


def test_duplicate_ids_are_rejected():
    chunk = Chunk("a:skills", "A", "skills", "t", "x")
    with pytest.raises(ValueError):
        KnowledgeBase([chunk, chunk])


@pytest.mark.parametrize("role", APP_ROLES)
@pytest.mark.parametrize(
    "topic,kinds",
    [
        ("required skills tools", ("skills",)),
        ("interview questions topics", ("interview",)),
        ("portfolio project ideas", ("projects",)),
        ("learning path roadmap", ("roadmap",)),
    ],
)
def test_every_app_role_retrieves_its_own_chunk(kb, role, topic, kinds):
    hits = kb.retrieve_for_role(role, topic, k=1, kinds=kinds)
    assert hits and hits[0].chunk.role == role and hits[0].chunk.kind == kinds[0]


@pytest.mark.parametrize(
    "role,expected",
    [
        ("Backend Developer", "Software Engineer"),
        ("Site Reliability Engineer", "DevOps Engineer"),
        ("Security Analyst", "Cybersecurity Engineer"),
        ("Business Analyst", "Data Analyst"),
    ],
)
def test_aliases_resolve(kb, role, expected):
    assert kb.matching_roles(role) == [expected]


@pytest.mark.parametrize("role", ["Quantum Engineer", "Product Manager", "Data Engineer", "UX Designer"])
def test_unknown_roles_get_no_reference_material(kb, role):
    assert kb.retrieve_for_role(role, "required skills tools") == []


def test_off_topic_query_abstains(kb):
    assert kb.retrieve("pastry chef culinary arts sourdough") == []


def test_kind_filter_is_respected(kb):
    hits = kb.retrieve("docker kubernetes pipeline", k=5, kinds=("projects",), min_score=0.0)
    assert hits and all(h.chunk.kind == "projects" for h in hits)


def test_mixed_signal_roles_keep_every_tied_candidate(kb):
    assert kb.matching_roles("Cloud Security Engineer") == ["Cloud Engineer", "Cybersecurity Engineer"]


def test_context_formatting_and_citation_filtering(kb):
    hits = kb.retrieve_for_role("Cloud Engineer", "required skills tools", k=2)
    context = format_context(hits)
    assert f"[{hits[0].chunk.id}]" in context
    assert cited_sources([hits[0].chunk.id, "made-up:id", hits[0].chunk.id], hits) == [hits[0].chunk.id]
    assert "none available" in format_context([])
