"""Standalone RAG CLI. Default demo is offline; real APIs require --allow-paid."""
import argparse
import json
from pathlib import Path
from ai_processor.providers import ProviderError
from config.rag_settings import RAGSettings
from rag.runtime import build_pipeline


class DemoProvider:
    """Deterministic mock, explicitly labelled; no model-quality claim."""
    def generate_response(self, messages):
        payload = json.loads(messages[-1]["content"])
        return json.dumps({"parts": [{"part_id": p["part_id"], "evidence": p["allowed_evidence"][:1]}
                                     for p in payload["parts"]]}, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--mode", choices=("fts", "vector", "hybrid"), default=None)
    parser.add_argument("--section")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--allow-paid", action="store_true")
    group.add_argument("--mock-llm", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if not (args.allow_paid or args.mock_llm):
        parser.error("Используйте --mock-llm для офлайн-проверки; --allow-paid только после разрешения")
    pipeline = None
    try:
        # Offline demo does not read secret files or require keys.
        settings = RAGSettings.from_env(env={}) if args.mock_llm else RAGSettings.from_env(env_file=root / ".env")
        pipeline = build_pipeline(root, settings, provider=DemoProvider() if args.mock_llm else None,
                                  allow_paid=args.allow_paid, mode=args.mode)
        result = pipeline.answer(args.question, mode=args.mode, section=args.section)
        result["llm_mode"] = "mock" if args.mock_llm else settings.llm_provider
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, ProviderError) as exc:
        # Configuration error messages are authored locally; no SDK exception text.
        print(json.dumps({"status": "configuration_error", "message": str(exc)}, ensure_ascii=False))
    except Exception:
        print(json.dumps({"status": "runtime_error", "message": "Не удалось запустить пайплайн; проверьте файлы индекса и настройки."}, ensure_ascii=False))
    finally:
        if pipeline is not None:
            provider = pipeline.provider
            if hasattr(provider, "close"):
                provider.close()
            elif hasattr(provider, "client"):
                provider.client.close()


if __name__ == "__main__":
    main()
