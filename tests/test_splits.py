from halluscope.data.schema import TOPICS, Item, Turn
from halluscope.data.splits import (
    assert_no_leakage,
    family_root,
    grouped_split,
    leave_one_topic_out,
)


def _fake_items(n_roots_per_topic=6, n_para=2):
    items = []
    for topic in TOPICS:
        for r in range(n_roots_per_topic):
            root = f"{topic}-{r:04d}"
            fams = [root] + [f"{root}-p{k}" for k in range(1, n_para + 1)]
            for fam in fams:
                for v, label in (("a", "specified"), ("b", "underspecified")):
                    items.append(
                        Item(
                            id=f"{fam}-{v}",
                            family=fam,
                            topic=topic,
                            variant=v,
                            label=label,
                            turns=[Turn(role="user", content=f"{fam} {v}")],
                            gap="g" if v == "b" else None,
                            expected_clarifying_question="q?" if v == "b" else None,
                        )
                    )
    return items


def test_family_root_strips_paraphrase_suffix():
    it = _fake_items(1, 1)[2]
    assert it.family.endswith("-p1")
    assert family_root(it) == it.family[: -len("-p1")]


def test_grouped_split_has_no_leakage_and_all_topics():
    items = _fake_items()
    splits = grouped_split(items, seed=3)
    assert_no_leakage(splits)
    for name in ("train", "val", "test"):
        assert {it.topic for it in splits[name]} == set(TOPICS)
    assert sum(len(v) for v in splits.values()) == len(items)


def test_grouped_split_is_deterministic():
    items = _fake_items()
    a = grouped_split(items, seed=7)
    b = grouped_split(items, seed=7)
    assert [i.id for i in a["test"]] == [i.id for i in b["test"]]


def test_leave_one_topic_out_yields_every_topic():
    items = _fake_items(2, 0)
    folds = list(leave_one_topic_out(items))
    assert [t for t, _, _ in folds] == sorted(TOPICS)
    for topic, train, test in folds:
        assert all(it.topic != topic for it in train)
        assert all(it.topic == topic for it in test)
