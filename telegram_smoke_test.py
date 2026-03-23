import argparse
import json
import os
import sys

from dotenv import load_dotenv

from backend.telegram_alerts import TelegramAlertError, fetch_telegram_updates, send_telegram_message


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Smoke-test Telegram alert delivery")
    parser.add_argument("--message", default="MarketScope test alert")
    parser.add_argument("--show-updates", action="store_true")
    args = parser.parse_args()

    try:
        if args.show_updates:
            updates = fetch_telegram_updates()
            print(json.dumps(updates, indent=2))
            return 0

        result = send_telegram_message(args.message)
        print(json.dumps({
            "ok": True,
            "message_id": ((result.get("result") or {}).get("message_id")),
            "chat": (((result.get("result") or {}).get("chat") or {}).get("id")),
        }, indent=2))
        return 0
    except TelegramAlertError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        return 1
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        return 1


if __name__ == "__main__":
    sys.exit(main())
