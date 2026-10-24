from types import SimpleNamespace

from app import ask


def test_ask_prints_token_usage_and_returns_response_text(monkeypatch, capsys):
    response = SimpleNamespace(
        usage_metadata=SimpleNamespace(prompt_token_count=12, candidates_token_count=34),
        text="回答",
    )
    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=lambda **kwargs: response)
    )
    monkeypatch.setattr(ask.genai, "Client", lambda **kwargs: client)
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")

    result = ask.ask("質問")

    assert result == "回答"
    assert "[トークン数] 入力: 12, 出力: 34" in capsys.readouterr().out