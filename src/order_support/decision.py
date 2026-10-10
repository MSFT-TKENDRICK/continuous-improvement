"""Synchronous client for Microsoft Foundry's Microsoft-Decision-1 API."""

from __future__ import annotations

import ipaddress
import json
import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from types import MappingProxyType
from typing import Any, Literal, Self, TypeAlias
from urllib.parse import urlsplit, urlunsplit

import httpx

SYSTEM_ONE_PATH = "/providers/microsoft/v1/systemone"
PROBABILITY_SUM_TOLERANCE = 1e-3
_QUESTION_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,127}$")
_RETRYABLE_STATUS_CODES = frozenset({429, *range(500, 600)})


class MicrosoftDecisionError(RuntimeError):
    """Raised when a Decision-1 request or response cannot be completed safely."""


def _require_nonempty_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _validate_question_name(name: Any) -> str:
    if not isinstance(name, str) or not _QUESTION_NAME.fullmatch(name):
        raise ValueError(
            "question names must start with a letter or underscore and contain "
            "only letters, digits, underscores, periods, or hyphens (maximum 128 characters)"
        )
    return name


def _validate_instructions(instructions: str | None) -> str | None:
    if instructions is None:
        return None
    return _require_nonempty_text(instructions, "instructions")


def _validate_criteria_description(value: Any, field: str) -> str | None:
    if value is None:
        return None
    return _require_nonempty_text(value, field)


@dataclass(frozen=True, slots=True)
class NoulQuestion:
    """A binary question whose answer is the probability of true."""

    instructions: str | None = None
    criteria: Mapping[str, str | None] | None = None
    type: Literal["noul"] = "noul"

    def __post_init__(self) -> None:
        instructions = _validate_instructions(self.instructions)
        criteria: Mapping[str, str | None] | None = None
        if self.criteria is not None:
            raw = dict(self.criteria)
            if not raw or not set(raw) <= {"true", "false"}:
                raise ValueError(
                    "noul criteria must contain only 'true' and/or 'false'"
                )
            criteria = MappingProxyType(
                {
                    key: _validate_criteria_description(value, f"criteria[{key!r}]")
                    for key, value in raw.items()
                }
            )
        if instructions is None and (
            criteria is None or all(value is None for value in criteria.values())
        ):
            raise ValueError(
                "noul questions require instructions or a criteria description"
            )
        object.__setattr__(self, "instructions", instructions)
        object.__setattr__(self, "criteria", criteria)

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"type": self.type}
        if self.instructions is not None:
            payload["instructions"] = self.instructions
        if self.criteria is not None:
            payload["criteria"] = dict(self.criteria)
        return payload


@dataclass(frozen=True, slots=True)
class ChoiceQuestion:
    """A categorical question selecting one named criterion."""

    criteria: Mapping[str, str | None]
    instructions: str | None = None
    type: Literal["choice"] = "choice"

    def __post_init__(self) -> None:
        raw = dict(self.criteria)
        if not 1 <= len(raw) <= 255:
            raise ValueError("choice criteria must contain between 1 and 255 options")
        criteria: dict[str, str | None] = {}
        for option, description in raw.items():
            option = _validate_question_name(option)
            criteria[option] = _validate_criteria_description(
                description, f"criteria[{option!r}]"
            )
        object.__setattr__(self, "criteria", MappingProxyType(criteria))
        object.__setattr__(
            self, "instructions", _validate_instructions(self.instructions)
        )

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": self.type,
            "criteria": dict(self.criteria),
        }
        if self.instructions is not None:
            payload["instructions"] = self.instructions
        return payload


@dataclass(frozen=True, slots=True)
class ScoreQuestion:
    """An ordinal question scored over criteria ordered from low to high."""

    criteria: Sequence[str]
    instructions: str | None = None
    type: Literal["score"] = "score"

    def __post_init__(self) -> None:
        criteria = tuple(self.criteria)
        if not 1 <= len(criteria) <= 10:
            raise ValueError("score criteria must contain between 1 and 10 levels")
        criteria = tuple(
            _require_nonempty_text(value, f"criteria[{index}]")
            for index, value in enumerate(criteria)
        )
        object.__setattr__(self, "criteria", criteria)
        object.__setattr__(
            self, "instructions", _validate_instructions(self.instructions)
        )

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": self.type,
            "criteria": list(self.criteria),
        }
        if self.instructions is not None:
            payload["instructions"] = self.instructions
        return payload


Question: TypeAlias = NoulQuestion | ChoiceQuestion | ScoreQuestion


@dataclass(frozen=True, slots=True)
class NoulAnswer:
    """Parsed binary answer."""

    probability: float

    @property
    def verdict(self) -> bool:
        return self.probability >= 0.5

    @property
    def confidence(self) -> float:
        return max(self.probability, 1.0 - self.probability)


@dataclass(frozen=True, slots=True)
class ChoiceAnswer:
    """Parsed categorical answer, preserving the service's probabilities."""

    choice: str
    probabilities: Mapping[str, float]
    confidence: float

    @property
    def verdict(self) -> str:
        return self.choice

    @property
    def modal_choice(self) -> str:
        return max(self.probabilities, key=self.probabilities.__getitem__)

    @property
    def modal_probability(self) -> float:
        return self.probabilities[self.modal_choice]


@dataclass(frozen=True, slots=True)
class ScoreAnswer:
    """Parsed ordinal answer, preserving its weighted score and distribution."""

    score: float
    legend: Mapping[int, str]
    probabilities: Mapping[int, float]
    confidence: float

    @property
    def modal_index(self) -> int:
        return max(self.probabilities, key=self.probabilities.__getitem__)

    @property
    def modal_label(self) -> str:
        return self.legend[self.modal_index]

    @property
    def verdict(self) -> str:
        return self.modal_label


Answer: TypeAlias = NoulAnswer | ChoiceAnswer | ScoreAnswer


@dataclass(frozen=True, slots=True)
class DecisionUsage:
    """Token usage reported by Decision-1."""

    input_tokens: int
    output_tokens: int

    def __add__(self, other: DecisionUsage) -> DecisionUsage:
        return DecisionUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


@dataclass(frozen=True, slots=True)
class DecisionProvenance:
    """Per-request provenance retained when questions are chunked."""

    request_index: int
    deployment: str
    underlying_model: str
    question_names: tuple[str, ...]
    attempts: int
    usage: DecisionUsage


@dataclass(frozen=True, slots=True)
class DecisionResult:
    """Aggregated answers and provenance for one logical decision request."""

    model: str
    answers: Mapping[str, Answer]
    usage: DecisionUsage
    provenance: tuple[DecisionProvenance, ...]


@dataclass(frozen=True, slots=True)
class _ParsedResponse:
    model: str
    answers: Mapping[str, Answer]
    usage: DecisionUsage


class MicrosoftDecisionClient:
    """Synchronous, retrying client for a Microsoft-Decision-1 deployment."""

    def __init__(
        self,
        endpoint: str,
        deployment: str,
        *,
        api_key: str | None = None,
        bearer_token: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 2,
        retry_backoff: float = 0.5,
        max_retry_delay: float = 30.0,
        max_questions_per_request: int = 64,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._endpoint = _normalize_endpoint(endpoint)
        self._url = f"{self._endpoint}{SYSTEM_ONE_PATH}"
        self._deployment = _require_nonempty_text(deployment, "deployment")
        if (api_key is None) == (bearer_token is None):
            raise ValueError("provide exactly one of api_key or bearer_token")
        credential = api_key if api_key is not None else bearer_token
        if not isinstance(credential, str) or not credential.strip():
            raise ValueError("the selected credential must be a non-empty string")
        if not _is_positive_finite(timeout):
            raise ValueError("timeout must be a positive finite number")
        if isinstance(max_retries, bool) or not isinstance(max_retries, int):
            raise TypeError("max_retries must be an integer")
        if not 0 <= max_retries <= 10:
            raise ValueError("max_retries must be between 0 and 10")
        if not _is_nonnegative_finite(retry_backoff):
            raise ValueError("retry_backoff must be a non-negative finite number")
        if not _is_nonnegative_finite(max_retry_delay):
            raise ValueError("max_retry_delay must be a non-negative finite number")
        if (
            isinstance(max_questions_per_request, bool)
            or not isinstance(max_questions_per_request, int)
            or not 1 <= max_questions_per_request <= 255
        ):
            raise ValueError("max_questions_per_request must be between 1 and 255")
        if not callable(sleep):
            raise TypeError("sleep must be callable")

        self._headers = (
            {"api-key": credential}
            if api_key is not None
            else {"Authorization": f"Bearer {credential}"}
        )
        self._headers["Accept"] = "application/json"
        self._max_retries = max_retries
        self._retry_backoff = float(retry_backoff)
        self._max_retry_delay = float(max_retry_delay)
        self._max_questions_per_request = max_questions_per_request
        self._sleep = sleep
        self._client = httpx.Client(timeout=float(timeout), transport=transport)
        self._closed = False

    @property
    def endpoint(self) -> str:
        return self._endpoint

    @property
    def deployment(self) -> str:
        return self._deployment

    def decide(
        self,
        state: str | Mapping[str, Any] | list[Any],
        questions: Mapping[str, Question],
    ) -> DecisionResult:
        """Evaluate state against one or more typed questions."""
        if self._closed:
            raise MicrosoftDecisionError("the Decision-1 client is closed")
        _validate_state(state)
        validated_questions = _validate_questions(questions)

        answers: dict[str, Answer] = {}
        usage = DecisionUsage(0, 0)
        provenance: list[DecisionProvenance] = []
        model: str | None = None
        items = list(validated_questions.items())

        for request_index, start in enumerate(
            range(0, len(items), self._max_questions_per_request)
        ):
            chunk = dict(items[start : start + self._max_questions_per_request])
            payload = {
                "model": self._deployment,
                "state": state,
                "questions": {
                    name: question.to_payload() for name, question in chunk.items()
                },
            }
            response, attempts = self._post(payload)
            parsed = _parse_response(response, chunk)
            if model is None:
                model = parsed.model
            elif parsed.model != model:
                raise MicrosoftDecisionError(
                    "Decision-1 returned inconsistent model identities across chunks"
                )
            answers.update(parsed.answers)
            usage += parsed.usage
            provenance.append(
                DecisionProvenance(
                    request_index=request_index,
                    deployment=self._deployment,
                    underlying_model=parsed.model,
                    question_names=tuple(chunk),
                    attempts=attempts,
                    usage=parsed.usage,
                )
            )

        return DecisionResult(
            model=model or "",
            answers=MappingProxyType(answers),
            usage=usage,
            provenance=tuple(provenance),
        )

    def _post(self, payload: Mapping[str, Any]) -> tuple[dict[str, Any], int]:
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.post(
                    self._url,
                    headers=self._headers,
                    json=payload,
                )
            except httpx.TransportError:
                if attempt == self._max_retries:
                    raise MicrosoftDecisionError(
                        f"Decision-1 transport failed after {attempt + 1} attempt(s)"
                    ) from None
                self._sleep(self._retry_delay(attempt, None))
                continue

            if response.status_code in _RETRYABLE_STATUS_CODES:
                if attempt == self._max_retries:
                    raise MicrosoftDecisionError(
                        f"Decision-1 returned HTTP {response.status_code} "
                        f"after {attempt + 1} attempt(s)"
                    )
                self._sleep(
                    self._retry_delay(attempt, response.headers.get("Retry-After"))
                )
                continue
            if not 200 <= response.status_code < 300:
                raise MicrosoftDecisionError(
                    f"Decision-1 returned HTTP {response.status_code}"
                )
            try:
                body = response.json()
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
                raise MicrosoftDecisionError(
                    "Decision-1 returned invalid JSON"
                ) from None
            if not isinstance(body, dict):
                raise MicrosoftDecisionError(
                    "Decision-1 response must be a JSON object"
                )
            return body, attempt + 1
        raise AssertionError("retry loop exited unexpectedly")

    def _retry_delay(self, attempt: int, retry_after: str | None) -> float:
        parsed = _parse_retry_after(retry_after)
        delay = parsed if parsed is not None else self._retry_backoff * (2**attempt)
        return min(delay, self._max_retry_delay)

    def close(self) -> None:
        if not self._closed:
            self._client.close()
            self._closed = True

    def __enter__(self) -> Self:
        if self._closed:
            raise MicrosoftDecisionError("the Decision-1 client is closed")
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _is_loopback_host(hostname: str | None) -> bool:
    if not hostname:
        return False
    if hostname.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _normalize_endpoint(endpoint: str) -> str:
    endpoint = _require_nonempty_text(endpoint, "endpoint").strip().rstrip("/")
    if endpoint.endswith(SYSTEM_ONE_PATH):
        endpoint = endpoint[: -len(SYSTEM_ONE_PATH)].rstrip("/")
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "endpoint must be an absolute HTTP(S) URL without credentials, query, or fragment"
        )
    if parsed.scheme == "http" and not _is_loopback_host(parsed.hostname):
        raise ValueError(
            "endpoint must use https; plain http is allowed only for loopback hosts "
            "(localhost, 127.0.0.0/8, ::1) so credentials are never sent unencrypted"
        )
    normalized_path = parsed.path.rstrip("/")
    return urlunsplit((parsed.scheme.lower(), parsed.netloc, normalized_path, "", ""))


def _is_positive_finite(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value > 0
    )


def _is_nonnegative_finite(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value >= 0
    )


def _validate_state(state: Any) -> None:
    if isinstance(state, str):
        if not state.strip():
            raise ValueError("state text must not be empty")
    elif isinstance(state, (dict, list)):
        _validate_json_value(state, "state", set())
    else:
        raise TypeError("state must be non-empty text, a JSON object, or a JSON array")


def _validate_json_value(value: Any, path: str, seen: set[int]) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return
    if isinstance(value, (dict, list)):
        identity = id(value)
        if identity in seen:
            raise ValueError(f"{path} contains a circular reference")
        seen.add(identity)
        try:
            if isinstance(value, dict):
                for key, item in value.items():
                    if not isinstance(key, str):
                        raise TypeError(f"{path} object keys must be strings")
                    _validate_json_value(item, f"{path}.{key}", seen)
            else:
                for index, item in enumerate(value):
                    _validate_json_value(item, f"{path}[{index}]", seen)
        finally:
            seen.remove(identity)
        return
    raise ValueError(f"{path} contains a non-JSON value of type {type(value).__name__}")


def _validate_questions(questions: Mapping[str, Question]) -> dict[str, Question]:
    if not isinstance(questions, Mapping) or not questions:
        raise ValueError("questions must be a non-empty mapping")
    validated: dict[str, Question] = {}
    for name, question in questions.items():
        name = _validate_question_name(name)
        if not isinstance(question, (NoulQuestion, ChoiceQuestion, ScoreQuestion)):
            raise TypeError(f"question {name!r} has an unsupported definition")
        validated[name] = question
    return validated


def _parse_response(
    payload: Mapping[str, Any],
    questions: Mapping[str, Question],
) -> _ParsedResponse:
    model = payload.get("model")
    if not isinstance(model, str) or not model.strip():
        raise MicrosoftDecisionError("Decision-1 response has an invalid model")
    raw_answers = payload.get("answers")
    if not isinstance(raw_answers, dict):
        raise MicrosoftDecisionError(
            "Decision-1 response has an invalid answers object"
        )
    expected = set(questions)
    actual = set(raw_answers)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        details = []
        if missing:
            details.append(f"missing {missing!r}")
        if unexpected:
            details.append(f"unexpected {unexpected!r}")
        raise MicrosoftDecisionError(
            "Decision-1 response answer names do not match the request: "
            + ", ".join(details)
        )

    answers: dict[str, Answer] = {}
    for name, question in questions.items():
        raw = raw_answers[name]
        if not isinstance(raw, dict):
            raise MicrosoftDecisionError(f"answer {name!r} must be an object")
        answer_type = raw.get("type")
        if answer_type != question.type:
            raise MicrosoftDecisionError(
                f"answer {name!r} has type {answer_type!r}; expected {question.type!r}"
            )
        if isinstance(question, NoulQuestion):
            answers[name] = _parse_noul(name, raw)
        elif isinstance(question, ChoiceQuestion):
            answers[name] = _parse_choice(name, raw, question)
        else:
            answers[name] = _parse_score(name, raw, question)

    return _ParsedResponse(
        model=model,
        answers=MappingProxyType(answers),
        usage=_parse_usage(payload.get("usage")),
    )


def _parse_noul(name: str, raw: Mapping[str, Any]) -> NoulAnswer:
    return NoulAnswer(
        probability=_probability(raw.get("noul"), f"answer {name!r}.noul")
    )


def _parse_choice(
    name: str,
    raw: Mapping[str, Any],
    question: ChoiceQuestion,
) -> ChoiceAnswer:
    choice = raw.get("choice")
    if not isinstance(choice, str) or choice not in question.criteria:
        raise MicrosoftDecisionError(f"answer {name!r} selected an unknown choice")
    probabilities = _probability_mapping(
        raw.get("probabilities"),
        expected_keys=set(question.criteria),
        field=f"answer {name!r}.probabilities",
    )
    return ChoiceAnswer(
        choice=choice,
        probabilities=MappingProxyType(probabilities),
        confidence=_probability(raw.get("confidence"), f"answer {name!r}.confidence"),
    )


def _parse_score(
    name: str,
    raw: Mapping[str, Any],
    question: ScoreQuestion,
) -> ScoreAnswer:
    score = _finite_number(raw.get("score"), f"answer {name!r}.score")
    maximum = len(question.criteria) - 1
    if not 0 <= score <= maximum:
        raise MicrosoftDecisionError(
            f"answer {name!r}.score must be between 0 and {maximum}"
        )
    expected_keys = {str(index) for index in range(len(question.criteria))}
    raw_legend = raw.get("legend")
    if not isinstance(raw_legend, dict) or set(raw_legend) != expected_keys:
        raise MicrosoftDecisionError(
            f"answer {name!r}.legend must contain exactly {sorted(expected_keys)!r}"
        )
    legend: dict[int, str] = {}
    for index, expected_label in enumerate(question.criteria):
        value = raw_legend[str(index)]
        if value != expected_label:
            raise MicrosoftDecisionError(
                f"answer {name!r}.legend does not match the requested criteria"
            )
        legend[index] = value
    raw_probabilities = _probability_mapping(
        raw.get("probabilities"),
        expected_keys=expected_keys,
        field=f"answer {name!r}.probabilities",
    )
    probabilities = {int(key): value for key, value in raw_probabilities.items()}
    return ScoreAnswer(
        score=score,
        legend=MappingProxyType(legend),
        probabilities=MappingProxyType(probabilities),
        confidence=_probability(raw.get("confidence"), f"answer {name!r}.confidence"),
    )


def _parse_usage(raw: Any) -> DecisionUsage:
    if not isinstance(raw, dict):
        raise MicrosoftDecisionError("Decision-1 response has invalid usage")
    return DecisionUsage(
        input_tokens=_token_count(raw.get("input_tokens"), "usage.input_tokens"),
        output_tokens=_token_count(raw.get("output_tokens"), "usage.output_tokens"),
    )


def _token_count(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MicrosoftDecisionError(f"{field} must be a non-negative integer")
    return value


def _finite_number(value: Any, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise MicrosoftDecisionError(f"{field} must be a finite number")
    return float(value)


def _probability(value: Any, field: str) -> float:
    value = _finite_number(value, field)
    if not 0 <= value <= 1:
        raise MicrosoftDecisionError(f"{field} must be between 0 and 1")
    return value


def _probability_mapping(
    raw: Any,
    *,
    expected_keys: set[str],
    field: str,
) -> dict[str, float]:
    if not isinstance(raw, dict) or set(raw) != expected_keys:
        raise MicrosoftDecisionError(
            f"{field} must contain exactly {sorted(expected_keys)!r}"
        )
    probabilities = {
        key: _probability(value, f"{field}[{key!r}]") for key, value in raw.items()
    }
    total = math.fsum(probabilities.values())
    if not math.isclose(
        total,
        1.0,
        rel_tol=0.0,
        abs_tol=PROBABILITY_SUM_TOLERANCE,
    ):
        raise MicrosoftDecisionError(f"{field} must sum to approximately 1")
    return probabilities


def _parse_retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        delay = float(value)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=UTC)
        delay = (retry_at - datetime.now(UTC)).total_seconds()
    if not math.isfinite(delay):
        return None
    return max(0.0, delay)
