"""Task-specific problem questions must produce words and assumptions the advisor actually uses."""

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from tools.mcp_advisor.advisor import _TASK_KEYWORDS, MISSING_CAPABILITIES  # noqa: E402
from tools.mcp_advisor.problem_schema import DOMAINS, QUESTIONS, compose_problem, problem_questions  # noqa: E402

CATALOG = Path(__file__).resolve().parents[1] / "tools" / "mcp_advisor" / "catalog.yaml"
GUARDRAIL_WORDS = ("periodic", "shock", "discontinu", "contact", "velocity", "stress", "displacement", "strain")
TASKS = ("surrogate_simulation", "pde_operator", "segmentation_classification", "field_fitting",
         "shape_completion", "object_detection", "registration")


def _advisor_words() -> set[str]:
    catalog = yaml.safe_load(CATALOG.read_text(encoding="utf-8"))
    entries = catalog.get("models", catalog) if isinstance(catalog, dict) else {}
    words = {kw for entry in entries.values() if isinstance(entry, dict) for kw in entry.get("keywords", [])}
    words |= {w for group in _TASK_KEYWORDS.values() for w in group}
    words |= {w for info in MISSING_CAPABILITIES.values() for w in info["keywords"]}
    return words | set(GUARDRAIL_WORDS)


def _phrases():
    yield from DOMAINS.values()
    for questions in QUESTIONS.values():
        for q in questions:
            yield from (o["keywords"] for o in q["options"].values()) if q["kind"] == "choice" else [q["keywords"]]


@pytest.mark.parametrize("phrase", sorted({p for p in _phrases() if p}))
def test_every_option_hits_an_advisor_keyword(phrase):
    assert any(word in phrase for word in _advisor_words()), f"{phrase!r} matches nothing in the advisor"


def test_domains_never_force_no_fit():
    blocked = [w for info in MISSING_CAPABILITIES.values() for w in info["keywords"]]
    for label, phrase in DOMAINS.items():
        assert not any(w in phrase for w in blocked), f"{label} would make the advisor answer no_fit"


@pytest.mark.parametrize("task", TASKS)
def test_every_task_has_questions_and_defaults_compose(task):
    schema = problem_questions(task)
    assert schema["questions"] and schema["domains"]
    assert compose_problem(task)["description"].startswith(task.replace("_", " "))


def test_segmentation_is_not_asked_about_time():
    keys = {q["key"] for q in problem_questions("segmentation_classification")["questions"]}
    assert "time" not in keys and {"granularity", "scale"} <= keys


def test_answers_set_assumptions():
    out = compose_problem("surrogate_simulation", {
        "target": "Vector field (velocity, displacement)", "geometry": "Arbitrary irregular shapes",
        "topology_changes": True,
    }, ["Physics · Fluid dynamics / CFD"], " airfoil ")
    assert out["assumptions"]["target_kind"] == "vector" and out["assumptions"]["geometry_varies"]
    assert out["dynamic_topology"] and out["description"].endswith("airfoil")
    rot = compose_problem("segmentation_classification", {"orientation": "Arbitrary rotations"})
    assert rot["assumptions"]["add_symmetries"] == ["rotation"]


def test_bad_answers_are_rejected():
    with pytest.raises(ValueError):
        compose_problem("registration", {"time": "Transient (rollout over time)"})
    with pytest.raises(ValueError):
        compose_problem("registration", {"transform": "Sideways"})


def test_instance_answer_reaches_no_fit():
    from tools.mcp_advisor.advisor import Advisor

    text = compose_problem("segmentation_classification", {"granularity": "Separate id per object instance"})
    assert Advisor().recommend_architecture(text["description"], "point_cloud", 4096)["verdict"] == "no_fit"
