import json
from datetime import datetime
from types import SimpleNamespace

from app import ask


def test_ask_prints_token_usage_and_returns_response_text(monkeypatch, capsys):
    response = SimpleNamespace(
        usage_metadata=SimpleNamespace(
            prompt_token_count=12,
            candidates_token_count=34,
            tool_use_prompt_token_count=0,
            thoughts_token_count=2,
            total_token_count=48,
        ),
        text="回答",
    )
    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=lambda **_: response)
    )
    monkeypatch.setattr(ask.genai, "Client", lambda **kwargs: client)
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")

    result = ask.ask("質問")

    assert result == "回答"
    assert "[トークン数] 入力: 12, 出力: 34" in capsys.readouterr().out


def test_ask_appends_comparable_jsonl_metrics(monkeypatch, tmp_path):
    response = SimpleNamespace(
        usage_metadata=SimpleNamespace(
            prompt_token_count=12,
            candidates_token_count=34,
            tool_use_prompt_token_count=0,
            thoughts_token_count=2,
            total_token_count=48,
        ),
        text="回答",
    )
    def generate_content(**kwargs):
        assert kwargs == {"model": ask.MODEL, "contents": "質問"}
        return response

    client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))

    def create_client(**kwargs):
        assert kwargs == {"api_key": "test-key"}
        return client

    monkeypatch.setattr(ask.genai, "Client", create_client)
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")
    times = iter([10.0, 10.125])
    monkeypatch.setattr(ask, "perf_counter", lambda: next(times))
    metrics_file = tmp_path / "metrics.jsonl"

    assert ask.ask(
        "質問", metrics_file=metrics_file, session_id="session-1", request_number=2
    ) == "回答"

    record = json.loads(metrics_file.read_text(encoding="utf-8"))
    assert datetime.fromisoformat(record["timestamp_utc"]).tzinfo is not None
    assert {key: value for key, value in record.items() if key != "timestamp_utc"} == {
        "session_id": "session-1",
        "request_number": 2,
        "model": ask.MODEL,
        "mode": "single",
        "question": "質問",
        "latency_ms": 125.0,
        "status": "success",
        "http_status": 200,
        "error_type": None,
        "input_tokens": 12,
        "output_tokens": 34,
        "tool_use_prompt_tokens": 0,
        "thoughts_tokens": 2,
        "total_tokens": 48,
    }


def test_ask_records_api_errors_for_rate_limit_analysis(monkeypatch, tmp_path):
    def raise_rate_limit(**kwargs):
        assert kwargs == {"model": ask.MODEL, "contents": "質問"}
        raise ask.errors.APIError(429, {})

    client = SimpleNamespace(models=SimpleNamespace(generate_content=raise_rate_limit))

    def create_client(**kwargs):
        assert kwargs == {"api_key": "test-key"}
        return client

    monkeypatch.setattr(ask.genai, "Client", create_client)
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")
    times = iter([10.0, 10.5])
    monkeypatch.setattr(ask, "perf_counter", lambda: next(times))
    metrics_file = tmp_path / "metrics.jsonl"

    try:
        ask.ask("質問", metrics_file=metrics_file)
    except ask.errors.APIError as error:
        assert error.code == 429
    else:
        raise AssertionError("APIError should be re-raised")

    record = json.loads(metrics_file.read_text(encoding="utf-8"))
    assert record["status"] == "error"
    assert record["http_status"] == 429
    assert record["latency_ms"] == 500.0
    assert record["input_tokens"] is None


def test_ask_sends_message_to_existing_chat(capsys):
    response = SimpleNamespace(
        usage_metadata=SimpleNamespace(prompt_token_count=5, candidates_token_count=8),
        text="続きの回答",
    )
    sent = []
    chat = SimpleNamespace(send_message=lambda question: sent.append(question) or response)

    result = ask.ask("続けて質問", chat=chat)

    assert result == "続きの回答"
    assert sent == ["続けて質問"]
    assert "[トークン数] 入力: 5, 出力: 8" in capsys.readouterr().out


def test_interactive_session_reuses_chat_history(monkeypatch, capsys, tmp_path):
    sent = []
    response = SimpleNamespace(
        usage_metadata=None,
        text="回答",
    )
    chat = SimpleNamespace(send_message=lambda question: sent.append(question) or response)
    client = SimpleNamespace(chats=SimpleNamespace(create=lambda **kwargs: chat))
    monkeypatch.setattr(ask.genai, "Client", lambda **kwargs: client)
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")
    questions = iter(["最初の質問", "続きの質問", "exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(questions))

    metrics_file = tmp_path / "metrics.jsonl"
    ask.ask_interactively(metrics_file)

    assert sent == ["最初の質問", "続きの質問"]
    output = capsys.readouterr().out
    assert output.count("[トークン数] 入力: 不明, 出力: 不明") == 2
    records = [json.loads(line) for line in metrics_file.read_text(encoding="utf-8").splitlines()]
    assert [record["request_number"] for record in records] == [1, 2]
    assert all(record["mode"] == "interactive" for record in records)