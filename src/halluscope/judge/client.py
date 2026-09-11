"""Provider-agnostic LLM client for judging, augmentation, and the simulated user.

Every logical role (``judge_primary``, ``judge_secondary``, ``simulated_user``,
``augmenter``) maps to an ordered list of LiteLLM model strings. Calls walk
that list so a rate limit or outage on one provider falls through to the next.
Structured outputs are parsed into pydantic models with light JSON repair and
retries. Every call appends a cost row to a JSONL log.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import orjson
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from halluscope.config import JudgeCfg

load_dotenv()

T = TypeVar("T", bound=BaseModel)

CompletionFn = Callable[..., Any]


class JudgeError(RuntimeError):
    pass


def _default_completion() -> CompletionFn:
    import litellm

    litellm.drop_params = True
    litellm.suppress_debug_info = True
    return litellm.completion


def extract_json(text: str) -> dict:
    """Parse JSON from model output, tolerating code fences and leading prose."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise JudgeError(f"no JSON object in output: {text[:200]!r}")
    return json.loads(text[start : end + 1])


def _response_cost(resp: Any) -> float:
    hidden = getattr(resp, "_hidden_params", None) or {}
    cost = hidden.get("response_cost")
    if cost is not None:
        return float(cost)
    try:
        import litellm

        return float(litellm.completion_cost(completion_response=resp))
    except Exception:
        return 0.0


def _usage(resp: Any) -> tuple[int, int]:
    u = getattr(resp, "usage", None)
    if u is None:
        return 0, 0
    return int(getattr(u, "prompt_tokens", 0) or 0), int(getattr(u, "completion_tokens", 0) or 0)


def _is_permanent(e: Exception) -> bool:
    """Missing key, bad key, unknown model, or a spent account: retrying is pointless.

    Exhausted credits arrive as a rate-limit error, which the retry loop would
    otherwise treat as a passing squall and sleep through on every single call.
    Over a verification run that is minutes of waiting to rediscover the same
    empty wallet, so a spent account counts as permanent and the alias falls
    through to the next provider immediately.
    """
    name = type(e).__name__.lower()
    msg = str(e).lower()
    return (
        "authentication" in name
        or "notfound" in name
        or "api key" in msg
        or "api_key" in msg
        or "does not exist" in msg
        or "model_not_found" in msg
        or "insufficient_quota" in msg
        or "exceeded your current quota" in msg
        or "no credits remaining" in msg
        or "used all available credits" in msg
        or "spending limit" in msg
        or "billing" in msg
        or "credit balance" in msg
    )


def _is_rate_limit(e: Exception) -> bool:
    """A 429 that is worth waiting out rather than counting against the retry budget."""
    name = type(e).__name__.lower()
    msg = str(e).lower()
    if _is_permanent(e):  # a spent account also answers 429; that one is not worth waiting for
        return False
    return "ratelimit" in name or "429" in msg or "too many requests" in msg


# Free provider tiers answer a burst with 429 for minutes, not seconds. The
# ordinary retry budget is three attempts inside twelve seconds, which is long
# enough to fail and short enough to learn nothing, so rate limits get their own
# schedule and do not consume the budget meant for genuine errors.
RATE_LIMIT_BACKOFF = (5.0, 15.0, 30.0, 60.0, 60.0, 120.0)


class JudgeClient:
    def __init__(
        self,
        cfg: JudgeCfg,
        cost_log: Path | str = "logs/llm_cost.jsonl",
        completion_fn: CompletionFn | None = None,
    ):
        self.cfg = cfg
        self.cost_log = Path(cost_log)
        self._completion = completion_fn or _default_completion()
        self._spent = 0.0
        self._calls = 0
        self._dead_models: set[str] = set()  # no key / not found: skip for the session

    # ---- raw text ---------------------------------------------------------------------
    def text(
        self,
        alias: str,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        tag: str = "",
    ) -> str:
        if alias not in self.cfg.aliases:
            raise JudgeError(f"unknown alias {alias!r}; known: {sorted(self.cfg.aliases)}")
        acfg = self.cfg.aliases[alias]
        temp = acfg.temperature if temperature is None else temperature
        mt = acfg.max_tokens if max_tokens is None else max_tokens
        last_err: Exception | None = None
        for model in acfg.models:
            if model in self._dead_models:
                continue
            attempt = 0
            throttled = 0
            while attempt < self.cfg.max_retries:
                t0 = time.time()
                try:
                    kwargs: dict[str, Any] = {
                        "model": model,
                        "messages": messages,
                        "temperature": temp,
                        "max_tokens": mt,
                    }
                    if json_mode:
                        kwargs["response_format"] = {"type": "json_object"}
                    resp = self._completion(**kwargs)
                    content = resp.choices[0].message.content or ""
                    # A reasoning model can spend the whole budget thinking and
                    # return nothing. That is a failed call, not an empty answer:
                    # returning "" here would send the caller into three rounds of
                    # JSON repair on a string that was never going to parse.
                    if not content.strip():
                        raise JudgeError(f"{model} returned empty content (budget {mt} tokens)")
                    self._log(alias, model, resp, time.time() - t0, tag, ok=True)
                    return content
                except Exception as e:  # noqa: BLE001 - provider errors are heterogeneous
                    last_err = e
                    self._log(
                        alias, model, None, time.time() - t0, tag, ok=False, error=str(e)[:200]
                    )
                    if _is_permanent(e):
                        self._dead_models.add(model)
                        break
                    if _is_rate_limit(e) and throttled < len(RATE_LIMIT_BACKOFF):
                        # Waiting out a throttle is not a failed attempt, so the
                        # budget for real errors survives the wait.
                        time.sleep(RATE_LIMIT_BACKOFF[throttled])
                        throttled += 1
                        continue
                    attempt += 1
                    time.sleep(min(2.0 * attempt, 6.0))
        raise JudgeError(f"all models failed for alias {alias!r}: {last_err}")

    # ---- structured -------------------------------------------------------------------
    def complete(
        self,
        alias: str,
        messages: list[dict[str, str]],
        schema: type[T],
        temperature: float | None = None,
        max_tokens: int | None = None,
        tag: str = "",
    ) -> T:
        """Return a validated ``schema`` instance. Repairs fenced JSON, retries on bad shape."""
        # Shown as a filled-in shape rather than the JSON-schema fragment, which
        # some models copy back verbatim, titles and all, instead of answering.
        props = schema.model_json_schema().get("properties", {})
        shape = {
            k: f"<{v.get('type', 'value')}>"
            + (f" {v['description']}" if v.get("description") else "")
            for k, v in props.items()
        }
        sys_extra = (
            "Respond with a single JSON object and nothing else, with these keys filled in "
            f"with actual values (not a schema): {json.dumps(shape, indent=0)}"
        )
        msgs = list(messages)
        if msgs and msgs[0]["role"] == "system":
            msgs[0] = {"role": "system", "content": msgs[0]["content"] + "\n\n" + sys_extra}
        else:
            msgs.insert(0, {"role": "system", "content": sys_extra})

        last_err: Exception | None = None
        for _ in range(self.cfg.max_retries):
            raw = self.text(
                alias, msgs, temperature=temperature, max_tokens=max_tokens, json_mode=True, tag=tag
            )
            try:
                return schema.model_validate(extract_json(raw))
            except (ValidationError, JudgeError, json.JSONDecodeError) as e:
                last_err = e
                msgs = msgs + [
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": f"That was not valid. Error: {e}. Reply with only the JSON object.",
                    },
                ]
        raise JudgeError(f"could not obtain valid {schema.__name__}: {last_err}")

    # ---- accounting -------------------------------------------------------------------
    def _log(
        self, alias: str, model: str, resp: Any, seconds: float, tag: str, ok: bool, error: str = ""
    ) -> None:
        pt, ct = _usage(resp) if resp is not None else (0, 0)
        cost = _response_cost(resp) if resp is not None else 0.0
        self._spent += cost
        self._calls += 1
        row = {
            "ts": time.time(),
            "alias": alias,
            "model": model,
            "ok": ok,
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "cost_usd": cost,
            "seconds": round(seconds, 3),
            "tag": tag,
            "error": error,
        }
        self.cost_log.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cost_log, "ab") as fh:
            fh.write(orjson.dumps(row))
            fh.write(b"\n")

    def cost_summary(self) -> dict[str, float]:
        return {"calls": float(self._calls), "spent_usd": round(self._spent, 4)}
