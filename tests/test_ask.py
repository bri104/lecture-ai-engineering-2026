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


def test_interactive_session_reuses_chat_history(monkeypatch, capsys):
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

    ask.ask_interactively()

    assert sent == ["最初の質問", "続きの質問"]
    output = capsys.readouterr().out
    assert output.count("[トークン数] 入力: 不明, 出力: 不明") == 2