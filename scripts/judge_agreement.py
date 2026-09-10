"""Agreement between two verification judges on the same families.

Each verify report holds one structured verdict per augmented family. Two
reports from different model families answer the question the single-judge
number cannot: is a family rejected because it is broken, or because one
model reads the rubric a particular way. Cohen's kappa is reported on the
pass/fail decision and on each rubric field, and the families the judges
disagree on are listed so a person can look at them.

    python scripts/judge_agreement.py results/verify_report_minimal.json \\
        results/verify_report_minimal_nemotron.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from halluscope.data.verify import FamilyVerdict, passes

FIELDS = ("b_omits_gap", "c_is_consistent", "d_contradicts", "off_domain")


def kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    agree = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    chance = pa * pb + (1 - pa) * (1 - pb)
    return float("nan") if chance == 1 else (agree - chance) / (1 - chance)


def load(path: Path) -> dict[str, FamilyVerdict]:
    raw = json.loads(path.read_text(encoding="utf-8"))["verdicts"]
    return {f: FamilyVerdict.model_validate(v) for f, v in raw.items() if v}


def main(a_path: str, b_path: str) -> dict:
    a, b = load(Path(a_path)), load(Path(b_path))
    shared = sorted(set(a) & set(b))
    out: dict = {
        "judge_a": a_path,
        "judge_b": b_path,
        "n_shared": len(shared),
        "n_only_a": len(set(a) - set(b)),
        "n_only_b": len(set(b) - set(a)),
    }
    pa = [passes(a[f]) for f in shared]
    pb = [passes(b[f]) for f in shared]
    out["pass_rate_a"] = sum(pa) / len(shared) if shared else float("nan")
    out["pass_rate_b"] = sum(pb) / len(shared) if shared else float("nan")
    out["kappa_pass"] = kappa(pa, pb)
    out["kappa_by_field"] = {
        k: kappa([getattr(a[f], k) for f in shared], [getattr(b[f], k) for f in shared])
        for k in FIELDS
    }
    out["disagree_on_pass"] = [
        {"family": f, "a_passes": x, "b_passes": y}
        for f, x, y in zip(shared, pa, pb, strict=True)
        if x != y
    ]
    return out


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    r = main(sys.argv[1], sys.argv[2])
    print(
        f"shared families {r['n_shared']}  (only in a {r['n_only_a']}, only in b {r['n_only_b']})"
    )
    print(f"pass rate  a {r['pass_rate_a']:.3f}  b {r['pass_rate_b']:.3f}")
    print(f"kappa on pass/fail {r['kappa_pass']:.3f}")
    for k, v in r["kappa_by_field"].items():
        print(f"  {k:18} {v:.3f}")
    print(f"disagree on {len(r['disagree_on_pass'])} families")
    for d in r["disagree_on_pass"]:
        print(
            f"  {d['family']:22} a={'pass' if d['a_passes'] else 'fail'}  b={'pass' if d['b_passes'] else 'fail'}"
        )
    Path("results").mkdir(exist_ok=True)
    Path("results/judge_agreement_verify.json").write_text(
        json.dumps(r, indent=2), encoding="utf-8"
    )
    print("wrote results/judge_agreement_verify.json")
