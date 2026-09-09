"""Kite login helper: python -m kite_login

Prints the login URL, waits for you to paste back the request_token from the
redirect, exchanges it locally for an access token, and saves the session.

You perform the login. Kite issues a request_token only after an interactive
login with your password and 2FA, so no tool can authenticate as you
unattended - which is the correct design and not a limitation to work
around. This script never prints your secret or your access token.

The token expires around 6am the next day. Kite provides retail apps no
non-interactive refresh, so this is a daily manual step.
"""
from __future__ import annotations

import logging
import sys
import urllib.parse

import config
import kite_client


def _report_credentials() -> bool:
    """Print which credentials are configured; True when a login can proceed."""
    present = kite_client.credentials_present()
    print("Credentials found in this environment:")
    for label, key in (("API key", "api_key"), ("API secret", "api_secret")):
        state = "set" if present[key] else "MISSING"
        print(f"  {label:<12}{state}")
    if present["session_file"]:
        print(f"  {'Session':<12}{config.KITE_TOKEN_FILE.name} exists "
              f"(will be overwritten)")
    if present["api_key"] and present["api_secret"]:
        return True
    print()
    print("Set them in your environment, not in a file in this repo:")
    print(f'  setx {config.KITE_API_KEY_ENV} "your_api_key"')
    print(f'  setx {config.KITE_API_SECRET_ENV} "your_api_secret"')
    print("Then open a NEW terminal, because setx only affects new shells.")
    return False


def _extract_request_token(pasted: str) -> str:
    """Pull request_token out of a full redirect URL or a bare token."""
    text = (pasted or "").strip()
    if not text:
        return ""
    looks_like_url = text.lower().startswith(("http://", "https://"))
    if looks_like_url or "request_token" in text:
        parsed = urllib.parse.urlparse(text)
        for part in (parsed.query, parsed.fragment):
            values = urllib.parse.parse_qs(part).get("request_token")
            if values and values[0].strip():
                return values[0].strip()
        # A URL carrying no request_token is a failed or wrong redirect.
        # Returning the whole URL as if it were the token would fail the
        # exchange with an opaque error instead of naming the real problem.
        return ""
    return text


def main() -> int:
    """Run the interactive login."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    print(f"\n{'=' * 72}")
    print("ZERODHA KITE LOGIN")
    print("=" * 72)
    if not _report_credentials():
        return 1
    try:
        url = kite_client.login_url()
    except kite_client.KiteError as exc:
        print(f"\n{exc}")
        return 1
    print()
    print("1. Open this URL and log in:")
    print(f"   {url}")
    print()
    print("2. Zerodha redirects to your app's redirect URL. Copy either the")
    print("   whole redirected URL or just the request_token from it.")
    print()
    try:
        pasted = input("3. Paste it here: ")
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.")
        return 1
    token = _extract_request_token(pasted)
    if not token:
        print("No request_token found in what you pasted.")
        return 1
    try:
        session = kite_client.exchange_request_token(token)
    except kite_client.KiteError as exc:
        print(f"\nLogin failed: {exc}")
        print("A request_token is single-use and expires within minutes, so "
              "if you reused one, start again from step 1.")
        return 1
    restricted = kite_client.save_session(session)
    print()
    print(f"Logged in as {session.user_id or 'your account'}.")
    if restricted:
        print(f"Session saved to {config.KITE_TOKEN_FILE.name} - gitignored, "
              f"and restricted to your account.")
    else:
        print(f"Session saved to {config.KITE_TOKEN_FILE.name} - gitignored, "
              f"but the file permissions could NOT be restricted, so treat "
              f"it as readable by anyone with access to this machine.")
    try:
        who = kite_client.profile(session)
        print(f"Verified: {who.get('user_name', '')} "
              f"({', '.join(who.get('exchanges') or [])})")
    except kite_client.KiteError as exc:
        print(f"Saved, but the verification call failed: {exc}")
    print()
    print("The token expires around 6am tomorrow; run this again then.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
