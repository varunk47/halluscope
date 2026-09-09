import pytest
from pydantic import ValidationError

from halluscope.data.schema import Family, Item, Turn


def _item(variant, label, turns, **kw):
    base = dict(
        id=f"rag-0001-{variant}",
        family="rag-0001",
        topic="rag",
        variant=variant,
        label=label,
        turns=turns,
    )
    base.update(kw)
    return Item(**base)


def test_variant_b_must_be_underspecified():
    with pytest.raises(ValidationError):
        _item("b", "specified", [Turn(role="user", content="Set up the index.")])


def test_underspecified_needs_gap_and_question():
    with pytest.raises(ValidationError):
        _item("b", "underspecified", [Turn(role="user", content="Set up the index.")])
    it = _item(
        "b",
        "underspecified",
        [Turn(role="user", content="Set up the index.")],
        gap="embedding model missing",
        expected_clarifying_question="Which embedding model?",
    )
    assert it.is_flag_target


def test_multi_turn_variants_need_three_turns():
    with pytest.raises(ValidationError):
        _item("c", "specified", [Turn(role="user", content="only one turn")])


def test_dialogue_must_end_on_user():
    with pytest.raises(ValidationError):
        _item(
            "a",
            "specified",
            [Turn(role="user", content="hi"), Turn(role="assistant", content="hello")],
        )


def test_prefix_returns_up_to_nth_user_turn():
    it = _item(
        "c",
        "specified",
        [
            Turn(role="user", content="u1"),
            Turn(role="assistant", content="a1"),
            Turn(role="user", content="u2"),
        ],
    )
    assert [t.content for t in it.prefix(1)] == ["u1"]
    assert [t.content for t in it.prefix(2)] == ["u1", "a1", "u2"]
    assert it.n_user_turns == 2


def test_family_requires_all_four_variants():
    a = _item("a", "specified", [Turn(role="user", content="x")])
    with pytest.raises(ValidationError):
        Family(family="rag-0001", topic="rag", items=[a, a, a, a])
