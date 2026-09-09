import pytest

from halluscope.data.schema import Turn

pytestmark = pytest.mark.network  # downloads the tiny model on first run


@pytest.fixture(scope="module")
def tiny():
    from halluscope.config import Settings
    from halluscope.models.loader import load_model

    return load_model(Settings().models["tiny"], device="cpu")


def test_render_contains_user_text(tiny):
    from halluscope.models.chat import render_chat

    turns = [Turn(role="user", content="Chunk the docs at 512 tokens.")]
    s = render_chat(tiny.tokenizer, turns)
    assert "Chunk the docs at 512 tokens." in s
    assert tiny.n_layers > 0 and tiny.hidden > 0


def test_generate_returns_n_with_logprobs(tiny):
    from halluscope.models.chat import generate

    turns = [Turn(role="user", content="Say hello in three words.")]
    gens = generate(tiny, turns, max_new_tokens=8, temperature=0.8, n=2, seed=0)
    assert len(gens) == 2
    for g in gens:
        assert len(g.logprobs) == len(g.token_ids)
        assert all(lp <= 0 for lp in g.logprobs)
