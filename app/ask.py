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
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from google import genai
from google.genai.chats import Chat
from google.genai import errors
from google.genai.types import GenerateContentResponse

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"
DEFAULT_METRICS_FILE = Path("metrics.jsonl")


def _token_usage(response: GenerateContentResponse | None) -> dict[str, int | None]:
    usage = response.usage_metadata if response else None
    return {
        "input_tokens": usage.prompt_token_count if usage else None,
        "output_tokens": usage.candidates_token_count if usage else None,
        "tool_use_prompt_tokens": usage.tool_use_prompt_token_count if usage else None,
        "thoughts_tokens": usage.thoughts_token_count if usage else None,
        "total_tokens": usage.total_token_count if usage else None,
    }


def _print_usage(response: GenerateContentResponse, latency_ms: float) -> None:
    usage = response.usage_metadata
    input_tokens = usage.prompt_token_count if usage and usage.prompt_token_count is not None else "不明"
    output_tokens = (
        usage.candidates_token_count if usage and usage.candidates_token_count is not None else "不明"
    )
    print(f"[応答時間] {latency_ms / 1000:.2f}秒")
    print(f"[トークン数] 入力: {input_tokens}, 出力: {output_tokens}")


def _append_metrics(metrics_file: Path, record: dict[str, object]) -> None:
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with metrics_file.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def ask(
    question: str,
    chat: Chat | None = None,
    *,
    metrics_file: Path | None = None,
    session_id: str | None = None,
    request_number: int = 1,
) -> str | None:
    started_at = datetime.now(timezone.utc)
    start_time = perf_counter()
    mode = "interactive" if chat is not None else "single"

    try:
        if chat is None:
            client = genai.Client(api_key=get_api_key())
            response = client.models.generate_content(model=MODEL, contents=question)
        else:
            response = chat.send_message(question)
    except errors.APIError as error:
        latency_ms = round((perf_counter() - start_time) * 1000, 3)
        print(f"[応答時間] {latency_ms / 1000:.2f}秒")
        if metrics_file is not None:
            _append_metrics(
                metrics_file,
                {
                    "timestamp_utc": started_at.isoformat(),
                    "session_id": session_id or str(uuid4()),
                    "request_number": request_number,
                    "model": MODEL,
                    "mode": mode,
                    "question": question,
                    "latency_ms": latency_ms,
                    "status": "error",
                    "http_status": error.code,
                    "error_type": type(error).__name__,
                    **_token_usage(None),
                },
            )
        raise

    latency_ms = round((perf_counter() - start_time) * 1000, 3)
    _print_usage(response, latency_ms)
    if metrics_file is not None:
        _append_metrics(
            metrics_file,
            {
                "timestamp_utc": started_at.isoformat(),
                "session_id": session_id or str(uuid4()),
                "request_number": request_number,
                "model": MODEL,
                "mode": mode,
                "question": question,
                "latency_ms": latency_ms,
                "status": "success",
                "http_status": 200,
                "error_type": None,
                **_token_usage(response),
            },
        )
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


def ask_interactively(metrics_file: Path = DEFAULT_METRICS_FILE) -> None:
    _configure_line_editing()
    client = genai.Client(api_key=get_api_key())
    chat = client.chats.create(model=MODEL)
    session_id = str(uuid4())
    request_number = 0
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
        request_number += 1
        print(f"\n[質問] {question}")
        print(
            ask(
                question,
                chat=chat,
                metrics_file=metrics_file,
                session_id=session_id,
                request_number=request_number,
            )
        )
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument(
        "--interactive", "-i", action="store_true", help="会話履歴を保って続けて質問する"
    )
    parser.add_argument(
        "--metrics-file",
        type=Path,
        default=DEFAULT_METRICS_FILE,
        help="計測結果を追記するJSONLファイル（既定: metrics.jsonl）",
    )
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    print(f"[計測ファイル] {args.metrics_file}")
    if args.interactive:
        try:
            ask_interactively(args.metrics_file)
        except errors.APIError as e:
            print(api_error_message(e.code, e.message))
            raise SystemExit(1)
        return

    print(f"[質問] {args.question}\n")
    try:
        print(ask(args.question, metrics_file=args.metrics_file))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
