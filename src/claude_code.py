"""Thin async wrapper around the `claude` CLI for non-interactive LLM calls."""
from __future__ import annotations

import asyncio
import logging

from src.config import settings

logger = logging.getLogger(__name__)

# Set to True via CLI --verbose to print full prompts and responses
VERBOSE: bool = False


async def call_claude(prompt: str, system: str = "") -> str:
    """Run a prompt through Claude Code CLI and return the text response.

    Uses asyncio subprocess exec (not shell=True) so the prompt string is passed
    as a direct argument — no shell injection risk.
    """
    full_prompt = f"{system}\n\n{prompt}" if system else prompt
    cmd = ["claude", "-p", full_prompt, "--model", settings.claude_model]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        err = stderr.decode()[:300]
        raise RuntimeError(f"claude CLI exited {proc.returncode}: {err}")

    response = stdout.decode().strip()

    if VERBOSE:
        print(f"\n{'='*60}\nPROMPT SENT TO CLAUDE:\n{'='*60}\n{full_prompt}\n")
        print(f"\n{'='*60}\nCLAUDE RESPONSE:\n{'='*60}\n{response}\n")

    return response
