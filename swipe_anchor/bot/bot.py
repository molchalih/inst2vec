"""Telegram bot runtime: the access gate + Mini App launcher.

Flow:
    /start <welcome_code>  -> if payload matches SA_TG_WELCOME_CODE, register the
                              user with the backend (/tg/register, internal token)
                              and greet + show the Mini App button.
    /start  (no payload)   -> already-registered users get the open button;
                              everyone else gets the invite-only message.

Network calls to the backend use httpx (already a top-level dep). All user-facing
text lives in ``copy.py``; all gate/delay logic lives in ``helpers.py``.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import httpx
from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
    WebAppInfo,
)
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

from swipe_anchor.bot import copy
from swipe_anchor.bot.helpers import (
    build_invite_link,
    is_admin,
    is_valid_welcome,
    new_invite_code,
    typing_delay_seconds,
)
from swipe_anchor.bot.invites import InviteStore

log = logging.getLogger("swipe_anchor.bot")


@dataclass(frozen=True)
class BotConfig:
    bot_token: str
    webapp_url: str
    backend_base_url: str
    internal_token: str
    welcome_code: str
    admin_ids: frozenset[int] = frozenset()
    invites_path: str = "data/invites.json"


def _open_markup(webapp_url: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton(copy.OPEN_BUTTON, web_app=WebAppInfo(url=webapp_url))]]
    )


async def _send_progressive(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    text: str,
    **kwargs: object,
) -> None:
    """Show a TYPING action, pause proportional to length, then send the text."""
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    await asyncio.sleep(typing_delay_seconds(text))
    await context.bot.send_message(chat_id=chat_id, text=text, **kwargs)


async def _register_user(cfg: BotConfig, telegram_id: int, name: str) -> bool:
    """Provision (or reactivate) the user's AccessCode via the backend."""
    url = f"{cfg.backend_base_url.rstrip('/')}/tg/register"
    try:
        async with httpx.AsyncClient(timeout=10.0) as http:
            resp = await http.post(
                url,
                json={"telegram_id": telegram_id, "name": name},
                headers={"X-Internal-Token": cfg.internal_token},
            )
        return resp.status_code == 200
    except httpx.HTTPError as exc:
        log.warning("register failed tg_id=%s err=%s", telegram_id, exc)
        return False


async def _already_registered(cfg: BotConfig, telegram_id: int) -> bool:
    url = f"{cfg.backend_base_url.rstrip('/')}/tg/exists"
    try:
        async with httpx.AsyncClient(timeout=10.0) as http:
            resp = await http.post(
                url,
                json={"telegram_id": telegram_id},
                headers={"X-Internal-Token": cfg.internal_token},
            )
        return resp.status_code == 200 and bool(resp.json().get("exists"))
    except httpx.HTTPError:
        return False


async def _fetch_stats(cfg: BotConfig) -> dict | None:
    """Pull the aggregate collection snapshot from the backend (admin-only)."""
    url = f"{cfg.backend_base_url.rstrip('/')}/tg/stats"
    try:
        async with httpx.AsyncClient(timeout=20.0) as http:
            resp = await http.post(
                url, headers={"X-Internal-Token": cfg.internal_token}
            )
        if resp.status_code != 200:
            return None
        return resp.json()
    except httpx.HTTPError as exc:
        log.warning("stats fetch failed: %s", exc)
        return None


# How often the bot drains due "remind me later" reminders from the backend.
REMINDER_POLL_S = 120


async def _fetch_due_reminders(cfg: BotConfig) -> list[int]:
    """Claim the telegram ids whose reminders are due (backend marks them sent)."""
    url = f"{cfg.backend_base_url.rstrip('/')}/tg/due-reminders"
    try:
        async with httpx.AsyncClient(timeout=15.0) as http:
            resp = await http.post(
                url, headers={"X-Internal-Token": cfg.internal_token}
            )
        if resp.status_code != 200:
            return []
        return [int(t) for t in resp.json().get("telegram_ids", [])]
    except (httpx.HTTPError, ValueError) as exc:
        log.warning("due-reminders fetch failed: %s", exc)
        return []


async def _reminder_loop(app: Application, cfg: BotConfig) -> None:
    """Poll for due reminders and DM each person their come-back nudge."""
    await asyncio.sleep(15)  # let startup settle before the first poll
    while True:
        for telegram_id in await _fetch_due_reminders(cfg):
            try:
                await app.bot.send_message(
                    chat_id=telegram_id,
                    text=copy.REMINDER,
                    reply_markup=_open_markup(cfg.webapp_url),
                )
            except Exception as exc:  # blocked the bot / deleted chat — skip
                log.info("reminder send failed tid=%s: %s", telegram_id, exc)
        await asyncio.sleep(REMINDER_POLL_S)


def build_start_handler(cfg: BotConfig, invites: InviteStore):
    async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        msg = update.effective_message
        user = update.effective_user
        if msg is None or user is None:
            return
        chat_id = msg.chat_id
        name = (user.first_name or user.username or "there").strip()
        payload = context.args[0] if context.args else None

        if await _already_registered(cfg, user.id):
            await _send_progressive(
                context,
                chat_id,
                copy.WELCOME_BACK.format(name=name),
                reply_markup=_open_markup(cfg.webapp_url),
            )
            return

        # Admit on the master welcome code OR any active admin-issued invite link.
        via_invite = invites.is_valid(payload)
        if not (is_valid_welcome(payload, cfg.welcome_code) or via_invite):
            await _send_progressive(context, chat_id, copy.NO_INVITE)
            return

        if not await _register_user(cfg, user.id, name):
            await _send_progressive(context, chat_id, copy.REGISTER_FAILED)
            return

        # Count the join against the specific invite that admitted them.
        if via_invite and payload:
            invites.record_use(payload)

        try:
            await context.bot.send_sticker(
                chat_id=chat_id, sticker=copy.pick_sticker_id()
            )
        except Exception as exc:
            log.info("sticker send skipped: %s", exc)
        # Type out the how-to one line at a time (each preceded by a TYPING pause),
        # then the outro carries the "open the game" button.
        await _send_progressive(context, chat_id, copy.GREETING_INTRO.format(name=name))
        for step in copy.GREETING_STEPS:
            await _send_progressive(context, chat_id, step)
        await _send_progressive(
            context,
            chat_id,
            copy.GREETING_OUTRO,
            reply_markup=_open_markup(cfg.webapp_url),
        )

    return start


def build_myid_handler():
    async def myid(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user = update.effective_user
        msg = update.effective_message
        if user is None or msg is None:
            return
        await msg.reply_text(copy.MYID.format(id=user.id))

    return myid


def build_invite_handler(cfg: BotConfig, invites: InviteStore):
    async def invite(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user = update.effective_user
        msg = update.effective_message
        if user is None or msg is None:
            return
        if not is_admin(user.id, cfg.admin_ids):
            await msg.reply_text(copy.NOT_ADMIN)
            return
        label = " ".join(context.args).strip() if context.args else ""
        code = new_invite_code()
        invites.add(code, label)
        link = build_invite_link(context.bot.username or "", code)
        await msg.reply_text(
            copy.INVITE_CREATED.format(
                label=f" ({label})" if label else "", link=link
            ),
            disable_web_page_preview=True,
        )

    return invite


def build_invites_handler(cfg: BotConfig, invites: InviteStore):
    async def invites_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user = update.effective_user
        msg = update.effective_message
        if user is None or msg is None:
            return
        if not is_admin(user.id, cfg.admin_ids):
            await msg.reply_text(copy.NOT_ADMIN)
            return
        rows = invites.active()
        if not rows:
            await msg.reply_text(copy.INVITES_EMPTY)
            return
        username = context.bot.username or ""
        lines = []
        for inv in rows:
            tag = f" — {inv.label}" if inv.label else ""
            joined = f"{inv.uses} joined" if inv.uses != 1 else "1 joined"
            lines.append(f"{build_invite_link(username, inv.code)}{tag}  ({joined})")
        await msg.reply_text("\n\n".join(lines), disable_web_page_preview=True)

    return invites_cmd


def build_revoke_handler(cfg: BotConfig, invites: InviteStore):
    async def revoke(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user = update.effective_user
        msg = update.effective_message
        if user is None or msg is None:
            return
        if not is_admin(user.id, cfg.admin_ids):
            await msg.reply_text(copy.NOT_ADMIN)
            return
        if not context.args:
            await msg.reply_text(copy.INVITE_NOT_FOUND)
            return
        # Accept either the bare code or a pasted full deeplink.
        code = context.args[0].strip()
        if "start=" in code:
            code = code.split("start=", 1)[1]
        ok = invites.deactivate(code)
        await msg.reply_text(copy.INVITE_REVOKED if ok else copy.INVITE_NOT_FOUND)

    return revoke


def build_stats_handler(cfg: BotConfig):
    async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user = update.effective_user
        msg = update.effective_message
        if user is None or msg is None:
            return
        chat_id = msg.chat_id
        if not is_admin(user.id, cfg.admin_ids):
            await msg.reply_text(copy.NOT_ADMIN)
            return

        data = await _fetch_stats(cfg)
        if data is None:
            await msg.reply_text(copy.STATS_FAILED)
            return
        totals = data["totals"]
        if not totals["responses"] and not sum(totals["comparisons"].values()):
            await msg.reply_text(copy.STATS_EMPTY)
            return

        # Headline numbers first (typed out), then one chart per message.
        await _send_progressive(context, chat_id, copy.stats_overview(totals))

        from swipe_anchor.bot import charts

        async def _send_chart(png: bytes, caption: str) -> None:
            await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.UPLOAD_PHOTO)
            await context.bot.send_photo(chat_id=chat_id, photo=png, caption=caption)

        try:
            if data.get("response_times"):
                _send = charts.cumulative_chart(data["response_times"])
                await _send_chart(_send, copy.CHART_CUMULATIVE)
            if sum(totals["comparisons"].values()):
                await _send_chart(
                    charts.status_donut(totals["comparisons"]), copy.CHART_STATUS
                )
            if data.get("per_annotator"):
                await _send_chart(
                    charts.contributors_bar(data["per_annotator"]), copy.CHART_CONTRIB
                )
        except Exception as exc:  # a render hiccup shouldn't kill the whole command
            log.warning("stats chart render failed: %s", exc)

    return stats


def build_application(cfg: BotConfig) -> Application:
    invites = InviteStore(cfg.invites_path)

    async def _post_init(app: Application) -> None:
        # Long-lived daemon that drains due reminders. Started on the running loop
        # in post_init; the reference on bot_data keeps it from being GC'd.
        app.bot_data["_reminder_task"] = asyncio.create_task(
            _reminder_loop(app, cfg)
        )

    app = Application.builder().token(cfg.bot_token).post_init(_post_init).build()
    app.add_handler(CommandHandler("start", build_start_handler(cfg, invites)))
    app.add_handler(CommandHandler("myid", build_myid_handler()))
    app.add_handler(CommandHandler("invite", build_invite_handler(cfg, invites)))
    app.add_handler(CommandHandler("invites", build_invites_handler(cfg, invites)))
    app.add_handler(CommandHandler("revoke", build_revoke_handler(cfg, invites)))
    app.add_handler(CommandHandler("stats", build_stats_handler(cfg)))
    return app
