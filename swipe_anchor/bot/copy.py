"""All user-facing bot strings + sticker ids in one place (easy to swap/translate).

Voice: lowercase, casual, warm, minimal punctuation.

The sticker ids are real, reusable public Telegram ``file_id`` values (love +
cool sets). They are referenced by id, never re-uploaded. If any becomes invalid,
swap it here — the bot sends stickers best-effort so a stale id never blocks the
flow.
"""

from __future__ import annotations

import random

# Reusable public sticker file_ids (warm / love set).
LOVE_STICKERS: list[str] = [
    "CAACAgIAAxkBAAIIi2cIbwABzoglgSTvgzYYhXPJPr6qvwACUzUAAlXqMEjIEbhFqUnV4DYE",
    "CAACAgIAAxkBAAIIjWcIbw-1QNArDO4OWtFR9v2OO7QzAAIvAAMlu1EtfOzBdWmGja42BA",
    "CAACAgIAAxkBAAIIj2cIby2ZSW3CDlM9FbQpGq3YJsw0AAI-AAMlu1EtpSgdxDW0ukI2BA",
    "CAACAgIAAxkBAAIIkWcIb4r2WNa-a6jDRmIAAQj9nWJFTQACQhEAAjc6eUhBk1ZaSf7SeTYE",
    "CAACAgIAAxkBAAIIk2cIb5cpS56ZPtH7fTJkYf8lv7SSAAI9AANDD70UZq9P1w6AYNM2BA",
    "CAACAgIAAxkBAAIIlWcIb8Um0JqbShbJke1ph2TkFDkqAAItAANDD70UA30he7CxbLQ2BA",
    "CAACAgIAAxkBAAIIl2cIcAwD3EChNOGmEb0Ck_S_dLyAAAIGMQACCuygSrZwFh1ISw83NgQ",
    "CAACAgIAAxkBAAIImWcIcCgeUm3ApTSa0heM6MYflkClAAImMwAC9rP5SLsSmhVgb2JFNgQ",
    "CAACAgIAAxkBAAIIm2cIcC1opUFkjGR2avvKoPHaYyaeAAJMNwACri6wSeWcDnJxiXUiNgQ",
    "CAACAgIAAxkBAAIInWcIcEdhh2lhpHPsfwABTjHvGJZPBwACcQADPIpXGo_yzPS-YYiQNgQ",
    "CAACAgIAAxkBAAIIpWcIcHFLNH8YNCimm_vnVTS7IIhnAAIeQgACsvLJSqy03EbL4wefNgQ",
]

# Reusable public sticker file_ids (cool / thumbs-up set).
COOL_STICKERS: list[str] = [
    "CAACAgEAAxkBAAIIiWcIZPdHEZxm0XxMypuRIigYeSr1AALpAQACOA6CERG4SPQscaSNNgQ",
    "CAACAgIAAxkBAAIIn2cIcFaegW-9Et0SGVxybqZQblfBAAJnAQACPIpXGtpCegThA_DeNgQ",
    "CAACAgIAAxkBAAIIoWcIcFkzYwABNPltOrW0nNMd7MDsBwAC2zEAAkS_wUoiNVZ8znaN-DYE",
    "CAACAgIAAxkBAAIIp2cIcHhaRUc5Rm6eY1kJLaQRIQ1wAALvEwACLcLQS65Z7MFh-3tvNgQ",
]

ALL_STICKERS: list[str] = LOVE_STICKERS + COOL_STICKERS


def pick_sticker_id(*, rng: random.Random | None = None) -> str:
    """Pick a random sticker from the combined love + cool set."""
    rng = rng or random.Random()
    return rng.choice(ALL_STICKERS)


# --- message strings -------------------------------------------------------

# The greeting is sent as a little typed-out sequence: an intro, then each step
# revealed on its own with a "typing…" pause before it, then the outro carrying
# the open button. Keeps the hand-written feel and makes the how-to easy to read.
GREETING_INTRO = "hey {name} 💛 super quick, here's how it works"

GREETING_STEPS = (
    "1. you'll see 3 accounts",
    "2. if one of them is the odd one out — tap it, then swipe up ⬆️",
    "3. they might all be the same — then just tap skip",
)

GREETING_OUTRO = "that's the whole thing. tap below whenever you've got a minute 👇"

WELCOME_BACK = "welcome back {name} 💛 tap below to jump back in"

NO_INVITE = (
    "hey 👋 this one's invite-only for now\n\n"
    "you'll need the personal start link a friend sent you — open that and "
    "i'll let you in"
)

OPEN_BUTTON = "open the game"

REGISTER_FAILED = "hmm something hiccuped on my end — try /start again in a sec 🙏"

# Sent ~20h after someone taps "remind me later" in the rest nudge.
REMINDER = (
    "hey 💛 you asked me to nudge you — whenever you've got a minute, "
    "there are still a few sets waiting for your eye\n\n"
    "no pressure at all, tap below if you're up for it 👇"
)

# --- admin command strings -------------------------------------------------

MYID = "your telegram id is {id}"

NOT_ADMIN = "sorry, this one's admins only 🙈"

INVITE_CREATED = (
    "here's a fresh invite link{label} 👇\n\n"
    "{link}\n\n"
    "send it to whoever you want in. see all of them with /invites, "
    "or kill one with /revoke <link>"
)

INVITES_EMPTY = "no active invite links yet — make one with /invite (optionally /invite some label)"

INVITE_REVOKED = "done — that link won't let anyone new in 🔒"

INVITE_NOT_FOUND = "couldn't find an active link for that — check /invites"

# --- stats command ---------------------------------------------------------

STATS_FAILED = "couldn't reach the stats right now — try again in a sec 🙏"

STATS_EMPTY = "no datapoints yet — once people start judging, this'll fill up 📊"

CHART_CUMULATIVE = "📈 every judgment collected so far, over time"
CHART_STATUS = "🍩 where the comparisons stand"
CHART_CONTRIB = "🏆 who's been doing the work"


def stats_overview(totals: dict) -> str:
    """The headline numbers, typed out before the charts land."""
    c = totals["comparisons"]
    gold_seen = totals.get("gold_seen", 0)
    gold_correct = totals.get("gold_correct", 0)
    gold_acc = (gold_correct / gold_seen) if gold_seen else 0.0
    lines = [
        "📊 swipe-anchor — live stats",
        "",
        "datapoints",
        f"• {totals['responses']} judgments collected",
        f"• {totals['triplets']} ordinal constraints materialized",
        f"• {totals['annotators']} people contributing",
        "",
        "comparisons",
        f"• {totals['resolved']} resolved "
        f"({c.get('retired', 0)} clear · {c.get('ambiguous', 0)} too-close)",
        f"• {c.get('open', 0)} still open · {c.get('gold', 0)} gold checks",
        "",
        "quality",
        f"• {totals['mean_agreement'] * 100:.0f}% mean agreement",
        f"• {totals['mean_reliability']:.2f} mean reliability",
    ]
    if gold_seen:
        lines.append(f"• {gold_acc * 100:.0f}% gold accuracy ({gold_correct}/{gold_seen})")
    lines += ["", "charts incoming 👇"]
    return "\n".join(lines)
