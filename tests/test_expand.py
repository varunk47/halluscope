from halluscope.data.expand import ACKS, FamilySpec, expand


def _spec():
    return FamilySpec(
        family="rag-9999",
        topic="rag",
        setup="We index docs.",
        specified="Chunk at 512 with bge-m3 and cosine, top 8.",
        underspecified="Chunk and embed the docs.",
        gap="chunk size, model, metric, k missing",
        question="Which chunk size, model, metric, and k?",
        assumption="picks defaults",
        fill="We use bge-m3, cosine, 512 chunks, top 8.",
        final="Write the code.",
        contradiction="We cannot run embeddings anymore.",
        update="Also, log every query to a file.",
    )


def test_expand_produces_four_valid_variants():
    fam = expand(_spec())
    by_v = {i.variant: i for i in fam.items}
    assert by_v["a"].label == "specified" and len(by_v["a"].turns) == 1
    assert (
        by_v["b"].label == "underspecified"
        and by_v["b"].reference_specified_variant == "rag-9999-a"
    )
    assert by_v["c"].label == "specified" and len(by_v["c"].turns) == 3
    assert (
        by_v["d"].label == "inconsistent" and by_v["d"].reference_specified_variant == "rag-9999-c"
    )
    assert by_v["c"].turns[-1].content == "Also, log every query to a file. Write the code."
    assert by_v["d"].turns[-1].content == "We cannot run embeddings anymore. Write the code."


def test_variant_c_and_d_share_first_turn_and_ack_is_neutral():
    fam = expand(_spec())
    by_v = {i.variant: i for i in fam.items}
    assert by_v["c"].turns[0].content == by_v["d"].turns[0].content
    assert by_v["c"].turns[1].content in ACKS
    assert by_v["d"].turns[1].content in ACKS


def test_expand_is_deterministic():
    a = expand(_spec()).model_dump()
    b = expand(_spec()).model_dump()
    assert a == b
