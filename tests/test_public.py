from halluscope.data.public import _item


def test_public_items_bypass_family_checks():
    it = _item(
        "ambigqa-amb-1",
        "public_ambigqa",
        "Who is the richest club in the championship?",
        "underspecified",
    )
    assert it.source == "public" and it.is_flag_target and it.variant == "b"
    ok = _item(
        "ambigqa-clr-1", "public_ambigqa", "What year did the Berlin Wall fall?", "specified"
    )
    assert not ok.is_flag_target and ok.gap is None
