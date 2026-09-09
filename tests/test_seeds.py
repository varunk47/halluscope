import re
from pathlib import Path

from halluscope.data.io import load_seed_families
from halluscope.data.schema import TOPICS

SEEDS = Path(__file__).resolve().parents[1] / "src" / "halluscope" / "data" / "seeds"

BANNED = re.compile(
    r"\b(beam|pde|reynolds|stress|strain|viscosity|heat equation|cantilever|buckling|"
    r"navier|finite element|mesh|turbulen|young'?s modulus|materials science|"
    r"crystal|lattice|microscop)\b",
    re.IGNORECASE,
)


def test_seed_counts_and_structure():
    fams = load_seed_families(SEEDS)
    assert len(fams) == 64, f"expected 64 families, got {len(fams)}"
    per_topic = {t: 0 for t in TOPICS}
    for f in fams:
        per_topic[f.topic] += 1
    assert all(n == 8 for n in per_topic.values()), per_topic
    ids = [it.id for f in fams for it in f.items]
    assert len(ids) == 256
    assert len(set(ids)) == 256


def test_seeds_contain_no_computational_science_vocabulary():
    fams = load_seed_families(SEEDS)
    hits = []
    for f in fams:
        for it in f.items:
            text = " ".join(t.content for t in it.turns) + " " + (it.gap or "")
            m = BANNED.search(text)
            if m:
                hits.append((it.id, m.group(0)))
    assert not hits, hits


def test_variant_c_and_d_share_context_with_gap_filled_early():
    fams = load_seed_families(SEEDS)
    for f in fams:
        by_v = {it.variant: it for it in f.items}
        assert len(by_v["c"].turns) >= 3
        assert len(by_v["d"].turns) >= 3
        assert by_v["b"].reference_specified_variant == by_v["a"].id
        assert by_v["d"].reference_specified_variant in (by_v["a"].id, by_v["c"].id)
