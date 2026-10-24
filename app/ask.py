"""第1回: 素のLLMに質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする
    uv run python -m app.ask --interactive  # 会話履歴を保って続けて質問する

通常の1回の呼び出しは、ツールも会話履歴も持たない「素の」呼び出し。
--interactive では、同じチャットに質問を送って会話履歴を保つ。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse
import sys

from google import genai
from google.genai.chats import Chat
from google.genai import errors
from google.genai.types import GenerateContentResponse

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def _print_usage(response: GenerateContentResponse) -> None:
    usage = response.usage_metadata
    input_tokens = usage.prompt_token_count if usage and usage.prompt_token_count is not None else "不明"
    output_tokens = (
        usage.candidates_token_count if usage and usage.candidates_token_count is not None else "不明"
    )
    print(f"[トークン数] 入力: {input_tokens}, 出力: {output_tokens}")


def ask(question: str, chat: Chat | None = None) -> str | None:
    if chat is None:
        client = genai.Client(api_key=get_api_key())
        response = client.models.generate_content(model=MODEL, contents=question)
    else:
        response = chat.send_message(question)

    _print_usage(response)

    return response.text


def _configure_line_editing() -> None:
    if sys.platform == "win32":
        return

    import readline

    readline.parse_and_bind("set editing-mode emacs")
    readline.parse_and_bind(r'"\C-h": backward-delete-char')
    readline.parse_and_bind(r'"\C-?": backward-delete-char')
    readline.parse_and_bind(r'"\e[D": backward-char')
    readline.parse_and_bind(r'"\e[C": forward-char')


def ask_interactively() -> None:
    _configure_line_editing()
    client = genai.Client(api_key=get_api_key())
    chat = client.chats.create(model=MODEL)
    print("会話を開始します。終了するには exit または quit を入力してください。")
    while True:
        try:
            question = input("質問> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        print(f"\n[質問] {question}")
        print(ask(question, chat=chat))
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument(
        "--interactive", "-i", action="store_true", help="会話履歴を保って続けて質問する"
    )
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    if args.interactive:
        try:
            ask_interactively()
        except errors.APIError as e:
            print(api_error_message(e.code, e.message))
            raise SystemExit(1)
        return

    print(f"[質問] {args.question}\n")
    try:
        print(ask(args.question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
