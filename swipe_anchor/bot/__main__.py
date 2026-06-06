"""Run the Telegram bot.

    uv run --group bot python -m swipe_anchor.bot

The repo runs from its root, so ``load_dotenv()`` with no path loads the ROOT
.env — but the bot's secrets live in ``swipe_anchor/.env``. We therefore load
that file explicitly by path.

Env (all in ``swipe_anchor/.env``):
    TG_TOK                  Telegram bot token (same token the backend validates)
    SA_TG_WEBAPP_URL        public https URL the Mini App is served from
    SA_TG_INTERNAL_TOKEN    shared secret for bot->backend /tg/register calls
    SA_TG_WELCOME_CODE      the deeplink payload that admits a new user
    SA_TG_ADMIN_IDS         comma-separated telegram ids allowed to run admin
                            commands (/invite, /invites, /revoke); empty = none
    SA_TG_INVITES_PATH      where admin-issued invite links are stored
                            (default: data/invites.json, relative to the cwd)
    BACKEND_BASE_URL        backend origin (default http://localhost:8100)
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from swipe_anchor.bot.bot import BotConfig, build_application
from swipe_anchor.bot.helpers import parse_admin_ids

log = logging.getLogger("swipe_anchor.bot")

_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"


def _load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(_ENV_PATH)
    except ImportError:
        pass


def _config_from_env() -> BotConfig:
    token = os.environ.get("TG_TOK", "")
    if not token:
        print("TG_TOK is not set — refusing to start the bot", file=sys.stderr)
        raise SystemExit(2)
    welcome = os.environ.get("SA_TG_WELCOME_CODE", "")
    if not welcome:
        print("SA_TG_WELCOME_CODE is not set — refusing to start", file=sys.stderr)
        raise SystemExit(2)
    # The bot cannot complete its advertised flow without these: an empty internal
    # token is always rejected by the backend, and an empty WebApp URL produces a
    # launcher button that can't open the Mini App. Fail closed before /start.
    internal_token = os.environ.get("SA_TG_INTERNAL_TOKEN", "")
    if not internal_token:
        print("SA_TG_INTERNAL_TOKEN is not set — refusing to start", file=sys.stderr)
        raise SystemExit(2)
    webapp_url = os.environ.get("SA_TG_WEBAPP_URL", "")
    if not webapp_url:
        print("SA_TG_WEBAPP_URL is not set — refusing to start", file=sys.stderr)
        raise SystemExit(2)
    return BotConfig(
        bot_token=token,
        webapp_url=webapp_url,
        backend_base_url=os.environ.get("BACKEND_BASE_URL", "http://localhost:8100"),
        internal_token=internal_token,
        welcome_code=welcome,
        admin_ids=parse_admin_ids(os.environ.get("SA_TG_ADMIN_IDS")),
        invites_path=os.environ.get("SA_TG_INVITES_PATH") or "data/invites.json",
    )


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    # httpx logs every Bot API request at INFO — and the URL embeds the bot
    # token. Quiet it so the token never lands in the log file.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    _load_env()
    cfg = _config_from_env()
    app = build_application(cfg)
    log.info("starting swipe-anchor bot (webapp=%s)", cfg.webapp_url)
    app.run_polling()
    return 0


if __name__ == "__main__":
    sys.exit(main())
