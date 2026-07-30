#!/usr/bin/env python3
"""Build light and dark GitHub profile cards.

The script can run fully offline from cached data, or collect public GitHub
statistics with the GraphQL API. It intentionally uses only Python's standard
library so the GitHub Actions job does not need a dependency-install step.
"""

from __future__ import annotations

import argparse
import calendar
import json
from html import escape as html_escape
import os
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import quote
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from pathlib import Path
from typing import Any, Iterable
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent
GRAPHQL_URL = "https://api.github.com/graphql"
USER_AGENT = "configurable-github-profile-card/1.0"


class GitHubAPIError(RuntimeError):
    """Raised when GitHub cannot satisfy a GraphQL request."""


@dataclass(frozen=True)
class Theme:
    name: str
    background: str
    border: str
    text: str
    key: str
    value: str
    muted: str
    addition: str
    deletion: str
    shadow: str


THEMES = {
    "dark": Theme(
        name="dark",
        background="#161b22",
        border="#30363d",
        text="#c9d1d9",
        key="#ffa657",
        value="#a5d6ff",
        muted="#6e7681",
        addition="#3fb950",
        deletion="#f85149",
        shadow="#010409",
    ),
    "light": Theme(
        name="light",
        background="#ffffff",
        border="#d0d7de",
        text="#24292f",
        key="#953800",
        value="#0550ae",
        muted="#8c959f",
        addition="#1a7f37",
        deletion="#cf222e",
        shadow="#afb8c1",
    ),
}


class GitHubClient:
    """Small GraphQL client with bounded retry behavior."""

    def __init__(self, token: str, timeout: int = 30) -> None:
        if not token:
            raise ValueError("A GitHub token is required for online mode.")
        self.token = token
        self.timeout = timeout
        self.request_count = 0

    def query(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        payload = json.dumps({"query": query, "variables": variables}).encode("utf-8")
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        }

        last_error: Exception | None = None
        for attempt in range(3):
            request = urllib.request.Request(
                GRAPHQL_URL,
                data=payload,
                headers=headers,
                method="POST",
            )
            try:
                self.request_count += 1
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                if body.get("errors"):
                    messages = "; ".join(
                        str(item.get("message", "Unknown GraphQL error"))
                        for item in body["errors"]
                    )
                    raise GitHubAPIError(messages)
                data = body.get("data")
                if not isinstance(data, dict):
                    raise GitHubAPIError("GitHub returned no GraphQL data.")
                return data
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                last_error = GitHubAPIError(
                    f"GitHub HTTP {exc.code}: {detail[:500]}"
                )
                if exc.code not in {429, 500, 502, 503, 504}:
                    break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc

            if attempt < 2:
                time.sleep(2**attempt)

        raise GitHubAPIError(str(last_error or "Unknown GitHub API failure"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="profile.json",
        help="Path to the profile JSON file (default: profile.json).",
    )
    parser.add_argument(
        "--username",
        help="Override the GitHub username resolved from environment/config.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Do not call GitHub; use cached or placeholder statistics.",
    )
    parser.add_argument(
        "--skip-loc",
        action="store_true",
        help="Fetch profile statistics but skip the expensive LOC refresh.",
    )
    return parser.parse_args()


def load_json(path: Path, default: Any) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return default
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def resolve_username(config: dict[str, Any], override: str | None) -> str:
    candidates = [
        override,
        os.getenv("GITHUB_USERNAME"),
        os.getenv("GITHUB_REPOSITORY_OWNER"),
    ]
    repository = os.getenv("GITHUB_REPOSITORY", "")
    if "/" in repository:
        candidates.append(repository.split("/", 1)[0])
    candidates.append(str(config.get("github_username", "")))

    for candidate in candidates:
        if candidate and candidate.strip():
            return candidate.strip()
    return "your-username"


def resolve_token() -> str:
    for name in ("PROFILE_TOKEN", "ACCESS_TOKEN", "GITHUB_TOKEN"):
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def parse_github_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def add_months(value: datetime, months: int) -> datetime:
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return value.replace(year=year, month=month, day=day)


def human_duration(start: datetime, end: datetime | None = None) -> str:
    end = end or datetime.now(timezone.utc)
    if start > end:
        start = end

    months_total = (end.year - start.year) * 12 + end.month - start.month
    anchor = add_months(start, months_total)
    if anchor > end:
        months_total -= 1
        anchor = add_months(start, months_total)

    years, months = divmod(months_total, 12)
    days = (end - anchor).days

    def unit(number: int, label: str) -> str:
        return f"{number} {label}{'' if number == 1 else 's'}"

    return f"{unit(years, 'year')}, {unit(months, 'month')}, {unit(days, 'day')}"


def personal_age(config: dict[str, Any], now: datetime | None = None) -> str:
    """Return age from ``birth_date`` using the configured local date.

    The profile stores only a birth date, not a birth time. Therefore, the age
    changes once per calendar day at midnight in ``timezone``. Supplying
    ``now`` is useful for deterministic tests; production calls use the
    current time in the configured timezone.
    """
    raw_birth_date = str(config.get("birth_date", "")).strip()
    if not raw_birth_date:
        raise ValueError(
            "profile.json must define birth_date in YYYY-MM-DD format "
            "when the {age} field is used."
        )

    try:
        parsed_birth_date = datetime.strptime(raw_birth_date, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(
            f"Invalid birth_date {raw_birth_date!r}; expected YYYY-MM-DD."
        ) from exc

    timezone_name = str(config.get("timezone", "UTC")).strip() or "UTC"
    try:
        local_timezone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(
            f"Unknown timezone {timezone_name!r} in profile.json."
        ) from exc

    birth = parsed_birth_date.replace(tzinfo=local_timezone)
    current = now.astimezone(local_timezone) if now else datetime.now(local_timezone)
    today = current.replace(hour=0, minute=0, second=0, microsecond=0)
    if birth > today:
        raise ValueError("birth_date cannot be in the future.")

    return human_duration(birth, today)


def collect_overview(
    client: GitHubClient,
    username: str,
    include_forks: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    query = """
    query($login: String!, $cursor: String) {
      user(login: $login) {
        id
        login
        createdAt
        followers { totalCount }
        owned: repositories(
          first: 100,
          after: $cursor,
          ownerAffiliations: [OWNER]
        ) {
          nodes {
            nameWithOwner
            isFork
            stargazers { totalCount }
            defaultBranchRef {
              target {
                ... on Commit {
                  oid
                  history { totalCount }
                }
              }
            }
          }
          pageInfo { hasNextPage endCursor }
        }
        affiliated: repositories(
          first: 1,
          ownerAffiliations: [OWNER, COLLABORATOR, ORGANIZATION_MEMBER]
        ) {
          totalCount
        }
      }
    }
    """

    cursor: str | None = None
    repositories: list[dict[str, Any]] = []
    followers = 0
    affiliated = 0
    created_at = ""
    login = username
    user_id = ""

    while True:
        data = client.query(query, {"login": username, "cursor": cursor})
        user = data.get("user")
        if not user:
            raise GitHubAPIError(f"GitHub user '{username}' was not found.")

        followers = int(user["followers"]["totalCount"])
        affiliated = int(user["affiliated"]["totalCount"])
        created_at = str(user["createdAt"])
        login = str(user["login"])
        user_id = str(user["id"])

        connection = user["owned"]
        for node in connection.get("nodes") or []:
            if not node:
                continue
            if node.get("isFork") and not include_forks:
                continue

            target = ((node.get("defaultBranchRef") or {}).get("target") or {})
            history = target.get("history") or {}
            repositories.append(
                {
                    "name_with_owner": str(node["nameWithOwner"]),
                    "is_fork": bool(node.get("isFork")),
                    "stars": int((node.get("stargazers") or {}).get("totalCount", 0)),
                    "head_oid": str(target.get("oid") or ""),
                    "commit_count": int(history.get("totalCount") or 0),
                }
            )

        page_info = connection["pageInfo"]
        if not page_info.get("hasNextPage"):
            break
        cursor = page_info.get("endCursor")

    summary = {
        "user_id": user_id,
        "username": login,
        "created_at": created_at,
        "account_age": human_duration(parse_github_datetime(created_at)),
        "followers": followers,
        "repos": len(repositories),
        "affiliated": affiliated,
        "stars": sum(int(item["stars"]) for item in repositories),
    }
    return summary, repositories


def collect_commit_contributions(
    client: GitHubClient,
    username: str,
    created_at: datetime,
) -> int:
    query = """
    query($login: String!, $from: DateTime!, $to: DateTime!) {
      user(login: $login) {
        contributionsCollection(from: $from, to: $to) {
          totalCommitContributions
        }
      }
    }
    """

    now = datetime.now(timezone.utc)
    total = 0
    start = created_at
    # GitHub limits a contribution collection to roughly one year. Using
    # 364-day windows also avoids leap-year edge cases.
    while start <= now:
        end = min(start + timedelta(days=364, hours=23, minutes=59, seconds=59), now)
        data = client.query(
            query,
            {
                "login": username,
                "from": start.isoformat().replace("+00:00", "Z"),
                "to": end.isoformat().replace("+00:00", "Z"),
            },
        )
        user = data.get("user")
        if not user:
            raise GitHubAPIError(f"GitHub user '{username}' was not found.")
        collection = user["contributionsCollection"]
        total += int(collection.get("totalCommitContributions") or 0)
        start = end + timedelta(seconds=1)
    return total


def collect_repository_loc(
    client: GitHubClient,
    name_with_owner: str,
    username: str,
) -> dict[str, int]:
    owner, repository = name_with_owner.split("/", 1)
    query = """
    query($owner: String!, $name: String!, $cursor: String) {
      repository(owner: $owner, name: $name) {
        defaultBranchRef {
          target {
            ... on Commit {
              history(first: 100, after: $cursor) {
                edges {
                  node {
                    additions
                    deletions
                    author { user { login } }
                  }
                }
                pageInfo { hasNextPage endCursor }
              }
            }
          }
        }
      }
    }
    """

    cursor: str | None = None
    additions = 0
    deletions = 0
    commits = 0
    target_login = username.casefold()

    while True:
        data = client.query(
            query,
            {"owner": owner, "name": repository, "cursor": cursor},
        )
        repo = data.get("repository")
        target = (((repo or {}).get("defaultBranchRef") or {}).get("target") or {})
        history = target.get("history")
        if not history:
            return {"additions": 0, "deletions": 0, "commits": 0}

        for edge in history.get("edges") or []:
            node = (edge or {}).get("node") or {}
            author = ((node.get("author") or {}).get("user") or {}).get("login")
            if author and str(author).casefold() == target_login:
                additions += int(node.get("additions") or 0)
                deletions += int(node.get("deletions") or 0)
                commits += 1

        page_info = history["pageInfo"]
        if not page_info.get("hasNextPage"):
            break
        cursor = page_info.get("endCursor")

    return {"additions": additions, "deletions": deletions, "commits": commits}


def refresh_loc_cache(
    client: GitHubClient,
    username: str,
    repositories: Iterable[dict[str, Any]],
    cache: dict[str, Any],
) -> tuple[dict[str, int], dict[str, Any]]:
    repository_cache = dict(cache.get("repositories") or {})
    active_names: set[str] = set()

    for repository in repositories:
        name = str(repository["name_with_owner"])
        active_names.add(name)
        head_oid = str(repository.get("head_oid") or "")
        commit_count = int(repository.get("commit_count") or 0)
        previous = repository_cache.get(name) or {}

        if not head_oid:
            repository_cache[name] = {
                "head_oid": "",
                "commit_count": 0,
                "additions": 0,
                "deletions": 0,
                "commits": 0,
            }
            continue

        if previous.get("head_oid") == head_oid:
            continue

        print(f"Refreshing LOC cache: {name}")
        try:
            loc = collect_repository_loc(client, name, username)
            repository_cache[name] = {
                "head_oid": head_oid,
                "commit_count": commit_count,
                **loc,
            }
        except GitHubAPIError as exc:
            if previous:
                print(
                    f"Warning: keeping stale LOC cache for {name}: {exc}",
                    file=sys.stderr,
                )
            else:
                print(f"Warning: unable to count LOC for {name}: {exc}", file=sys.stderr)
                repository_cache[name] = {
                    "head_oid": "",
                    "commit_count": commit_count,
                    "additions": 0,
                    "deletions": 0,
                    "commits": 0,
                }

    repository_cache = {
        name: value for name, value in repository_cache.items() if name in active_names
    }
    additions = sum(int(item.get("additions") or 0) for item in repository_cache.values())
    deletions = sum(int(item.get("deletions") or 0) for item in repository_cache.values())
    commits = sum(int(item.get("commits") or 0) for item in repository_cache.values())

    cache["repositories"] = repository_cache
    return {
        "loc_add": additions,
        "loc_del": deletions,
        "loc_net": additions - deletions,
        "loc_commits": commits,
    }, cache


def placeholder_summary(username: str) -> dict[str, Any]:
    return {
        "username": username,
        "account_age": "pending first sync",
        "repos": "sync",
        "affiliated": "sync",
        "stars": "sync",
        "commits": "sync",
        "followers": "sync",
        "loc_add": "sync",
        "loc_del": "sync",
        "loc_net": "sync",
    }


def normalize_summary(summary: dict[str, Any], username: str) -> dict[str, Any]:
    result = placeholder_summary(username)
    result.update(summary)
    result["username"] = str(result.get("username") or username)
    return result


def collect_stats(
    config: dict[str, Any],
    username: str,
    offline: bool,
    skip_loc: bool,
) -> tuple[dict[str, Any], Path, int]:
    stats_config = config.get("stats") or {}
    cache_path = ROOT / str(stats_config.get("cache_file", "cache/stats_cache.json"))
    cache = load_json(cache_path, {"summary": {}, "repositories": {}})
    cached_summary = normalize_summary(dict(cache.get("summary") or {}), username)

    if offline:
        return cached_summary, cache_path, 0

    token = resolve_token()
    if not token:
        print(
            "Warning: no GitHub token found; using cached statistics. ",
            "Set PROFILE_TOKEN or GITHUB_TOKEN for live data.",
            file=sys.stderr,
        )
        return cached_summary, cache_path, 0

    client = GitHubClient(token)
    include_forks = bool(stats_config.get("include_forks", False))

    try:
        overview, repositories = collect_overview(client, username, include_forks)
        created_at = parse_github_datetime(str(overview["created_at"]))
        overview["commits"] = collect_commit_contributions(client, username, created_at)

        loc_enabled = bool(stats_config.get("collect_loc", True)) and not skip_loc
        if loc_enabled:
            loc_summary, cache = refresh_loc_cache(
                client,
                str(overview["username"]),
                repositories,
                cache,
            )
            overview.update(loc_summary)
        else:
            for key in ("loc_add", "loc_del", "loc_net"):
                overview[key] = cached_summary.get(key, "--")

        summary = normalize_summary(overview, username)
        cache["summary"] = summary
        cache["updated_at"] = datetime.now(timezone.utc).isoformat()
        save_json(cache_path, cache)
        return summary, cache_path, client.request_count
    except GitHubAPIError as exc:
        print(f"Warning: live GitHub update failed: {exc}", file=sys.stderr)
        return cached_summary, cache_path, client.request_count


def format_number(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float) and value.is_integer():
        return f"{int(value):,}"
    return str(value)


class SafeFormatDict(dict[str, str]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def apply_fields(value: Any, fields: dict[str, str]) -> str:
    return str(value).format_map(SafeFormatDict(fields))


def text_span(
    text: str,
    css_class: str | None = None,
    x: int | None = None,
    y: int | None = None,
) -> str:
    attributes: list[str] = []
    if css_class:
        attributes.append(f'class="{css_class}"')
    if x is not None:
        attributes.append(f'x="{x}"')
    if y is not None:
        attributes.append(f'y="{y}"')
    attr_text = (" " + " ".join(attributes)) if attributes else ""
    return f"<tspan{attr_text}>{escape(text)}</tspan>"


def leader_count(columns: int, label: str, value: str) -> int:
    prefix_length = len(f". {label}: ")
    return max(1, columns - prefix_length - len(value) - 1)


def render_regular_row(
    x: int,
    y: int,
    columns: int,
    label: str,
    value: str,
) -> str:
    dots = "." * leader_count(columns, label, value)
    return "".join(
        [
            text_span(". ", "muted", x=x, y=y),
            text_span(label, "key"),
            text_span(": "),
            text_span(f"{dots} ", "muted"),
            text_span(value, "value"),
        ]
    )


def render_header(x: int, y: int, columns: int, title: str) -> str:
    rule = "-" * max(3, columns - len(title) - 1)
    return text_span(title, x=x, y=y) + text_span(" " + rule, "rule")


def render_stats_line_one(
    x: int,
    y: int,
    columns: int,
    summary: dict[str, Any],
) -> str:
    repos = format_number(summary["repos"])
    contributed = format_number(summary["affiliated"])
    stars = format_number(summary["stars"])

    fixed = (
        len(". Repos: ")
        + 1
        + len(repos)
        + len(" {Contributed: ")
        + len(contributed)
        + len("} | Stars: ")
        + 1
        + len(stars)
    )
    available = max(2, columns - fixed)
    repo_dot_count = min(4, available - 1)
    star_dot_count = max(1, available - repo_dot_count)

    return "".join(
        [
            text_span(". ", "muted", x=x, y=y),
            text_span("Repos", "key"),
            text_span(": "),
            text_span("." * repo_dot_count + " ", "muted"),
            text_span(repos, "value"),
            text_span(" {"),
            text_span("Contributed", "key"),
            text_span(": "),
            text_span(contributed, "value"),
            text_span("} | "),
            text_span("Stars", "key"),
            text_span(": "),
            text_span("." * star_dot_count + " ", "muted"),
            text_span(stars, "value"),
        ]
    )


def render_stats_line_two(
    x: int,
    y: int,
    columns: int,
    summary: dict[str, Any],
) -> str:
    commits = format_number(summary["commits"])
    followers = format_number(summary["followers"])

    fixed = (
        len(". Commits: ")
        + 1
        + len(commits)
        + len(" | Followers: ")
        + 1
        + len(followers)
    )
    available = max(2, columns - fixed)
    target_separator_column = min(40, columns - 18)
    commit_dot_count = target_separator_column - (
        len(". Commits: ") + 1 + len(commits)
    )
    commit_dot_count = max(1, min(commit_dot_count, available - 1))
    follower_dot_count = max(1, available - commit_dot_count)

    return "".join(
        [
            text_span(". ", "muted", x=x, y=y),
            text_span("Commits", "key"),
            text_span(": "),
            text_span("." * commit_dot_count + " ", "muted"),
            text_span(commits, "value"),
            text_span(" | "),
            text_span("Followers", "key"),
            text_span(": "),
            text_span("." * follower_dot_count + " ", "muted"),
            text_span(followers, "value"),
        ]
    )


def render_loc_line(
    x: int,
    y: int,
    columns: int,
    summary: dict[str, Any],
) -> str:
    label = "Lines of Code on GitHub"
    values = (summary["loc_net"], summary["loc_add"], summary["loc_del"])
    if not all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in values
    ):
        return render_regular_row(x, y, columns, label, "pending first sync")

    net = format_number(summary["loc_net"])
    added = format_number(summary["loc_add"])
    deleted = format_number(summary["loc_del"])
    plain_value = f"{net} ( {added}++, {deleted}-- )"
    dots = "." * leader_count(columns, label, plain_value)
    return "".join(
        [
            text_span(". ", "muted", x=x, y=y),
            text_span(label, "key"),
            text_span(": "),
            text_span(dots + " ", "muted"),
            text_span(net, "value"),
            text_span(" ( "),
            text_span(added, "addition"),
            text_span("++", "addition"),
            text_span(", "),
            text_span(deleted, "deletion"),
            text_span("--", "deletion"),
            text_span(" )"),
        ]
    )



SOCIALS_START_MARKER = "<!-- PROFILE-SOCIALS:START -->"
SOCIALS_END_MARKER = "<!-- PROFILE-SOCIALS:END -->"


def normalize_hex_color(value: Any, fallback: str = "6e7681") -> str:
    """Return a six-digit hexadecimal color suitable for SVG and Shields.io."""
    color = str(value or "").strip().lstrip("#")
    if len(color) == 3 and all(character in "0123456789abcdefABCDEF" for character in color):
        color = "".join(character * 2 for character in color)
    if len(color) != 6 or any(character not in "0123456789abcdefABCDEF" for character in color):
        color = fallback
    return color.upper()


def configured_socials(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return normalized social entries while preserving their configured order."""
    normalized: list[dict[str, Any]] = []
    for index, raw_social in enumerate(config.get("socials") or []):
        if not isinstance(raw_social, dict):
            raise ValueError(f"Social entry {index + 1} must be a JSON object.")

        name = str(raw_social.get("name", "")).strip()
        if not name:
            raise ValueError(f"Social entry {index + 1} must define a name.")

        raw_url = str(raw_social.get("url", "")).strip()
        if raw_url and not raw_url.startswith(("https://", "http://", "mailto:")):
            raise ValueError(
                f"Social URL for {name!r} must begin with https://, http://, or mailto:."
            )

        normalized.append(
            {
                "name": name,
                "url": raw_url,
                "enabled": bool(raw_social.get("enabled", True)),
                "show_in_card": bool(raw_social.get("show_in_card", True)),
                "icon": str(raw_social.get("icon", name[:2])).strip() or name[:2],
                "logo": str(raw_social.get("logo", "")).strip(),
                "logo_color": str(raw_social.get("logo_color", "white")).strip() or "white",
                "color": normalize_hex_color(raw_social.get("color")),
            }
        )
    return normalized


def shields_badge_url(social: dict[str, Any]) -> str:
    """Build a stable Shields.io URL without depending on extra packages."""
    label = quote(str(social["name"]), safe="")
    color = quote(str(social["color"]), safe="")
    parameters = ["style=for-the-badge"]
    if social.get("logo"):
        parameters.append("logo=" + quote(str(social["logo"]), safe=""))
    if social.get("logo_color"):
        parameters.append("logoColor=" + quote(str(social["logo_color"]), safe=""))
    return f"https://img.shields.io/badge/{label}-{color}?" + "&".join(parameters)


def render_socials_readme_block(config: dict[str, Any]) -> str:
    """Render the real clickable social buttons used by GitHub README pages.

    The profile card itself is embedded with ``<img>``. Browsers intentionally do
    not expose links nested inside an SVG loaded as an image, so the dependable
    GitHub-compatible click targets must live in README HTML.
    """
    active_socials = [
        social
        for social in configured_socials(config)
        if social["enabled"] and social["url"]
    ]

    lines = [SOCIALS_START_MARKER, '<p align="center">']
    for social in active_socials:
        url = html_escape(str(social["url"]), quote=True)
        name = html_escape(str(social["name"]), quote=True)
        badge_url = html_escape(shields_badge_url(social), quote=True)
        lines.extend(
            [
                f'  <a href="{url}" target="_blank" rel="noopener noreferrer">',
                f'    <img src="{badge_url}" alt="{name}" />',
                "  </a>",
            ]
        )
    lines.extend(["</p>", SOCIALS_END_MARKER])
    return "\n".join(lines)


def update_readme_socials(config: dict[str, Any]) -> None:
    """Replace or append the generated clickable-social marker block."""
    if not bool(config.get("update_readme_socials", True)):
        return

    readme_path = ROOT / str(config.get("readme_file", "README.md"))
    try:
        existing = readme_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        existing = ""

    block = render_socials_readme_block(config)
    start = existing.find(SOCIALS_START_MARKER)
    end = existing.find(SOCIALS_END_MARKER)

    if start >= 0 and end >= start:
        end += len(SOCIALS_END_MARKER)
        updated = existing[:start].rstrip() + "\n\n" + block + existing[end:]
    else:
        updated = existing.rstrip() + ("\n\n" if existing.strip() else "") + block + "\n"

    if updated != existing:
        readme_path.write_text(updated, encoding="utf-8")


def estimate_social_badge_width(social: dict[str, Any], font_size: int) -> int:
    """Estimate a compact pixel width for one terminal-style social badge."""
    icon_width = max(1, len(str(social["icon"]))) * (font_size * 0.58)
    label_width = max(1, len(str(social["name"]))) * (font_size * 0.62)
    return int(round(22 + icon_width + 8 + label_width + 14))


def render_social_badges(
    x: int,
    y: int,
    max_width: int,
    height: int,
    font_size: int,
    gap: int,
    socials: list[dict[str, Any]],
) -> str:
    """Draw compact badge-like pills inside the unchanged profile-card canvas."""
    visible = [social for social in socials if social["show_in_card"]]
    if not visible:
        return ""

    widths = [estimate_social_badge_width(social, font_size) for social in visible]
    total = sum(widths) + gap * max(0, len(widths) - 1)

    # Keep every badge inside the existing right panel. The current five badges
    # fit without scaling; this fallback makes later custom labels safe too.
    if total > max_width:
        scale = max_width / total
        widths = [max(58, int(width * scale)) for width in widths]
        adjusted_total = sum(widths) + gap * max(0, len(widths) - 1)
        if adjusted_total > max_width:
            overflow = adjusted_total - max_width
            widths[-1] = max(52, widths[-1] - overflow)

    parts = ['<g class="social-badges" aria-label="Social profile buttons">']
    cursor = x
    baseline = y + int(round(height * 0.68))
    for social, width in zip(visible, widths):
        active = bool(social["enabled"] and social["url"])
        opacity = "1" if active else "0.42"
        fill = "#" + str(social["color"])
        icon = escape(str(social["icon"]))
        name = escape(str(social["name"]))
        parts.extend(
            [
                f'<g opacity="{opacity}">',
                f'<rect x="{cursor}" y="{y}" width="{width}" height="{height}" '
                f'rx="4" fill="{fill}"/>',
                f'<text class="social-badge-text" x="{cursor + 11}" y="{baseline}">',
                f'<tspan class="social-badge-icon">{icon}</tspan>',
                '<tspan dx="8">' + name + '</tspan>',
                "</text>",
                "</g>",
            ]
        )
        cursor += width + gap
    parts.append("</g>")
    return "".join(parts)


def load_portrait(
    text_path: Path,
    map_path: Path | None = None,
) -> list[list[tuple[str, float]]]:
    """Load portrait characters and optional per-character opacity values.

    ``portrait.txt`` remains the human-readable ASCII source. When
    ``portrait_map.json`` exists, its opacity map adds tonal detail without
    embedding or publishing the original photograph.
    """
    lines = text_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"Portrait file {text_path} is empty.")

    if map_path is None or not map_path.exists():
        return [[(character, 1.0) for character in line] for line in lines]

    raw_map = load_json(map_path, None)
    rows = raw_map.get("rows") if isinstance(raw_map, dict) else raw_map
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Portrait map {map_path} must contain a non-empty rows list.")

    portrait: list[list[tuple[str, float]]] = []
    for row_number, raw_row in enumerate(rows, start=1):
        if not isinstance(raw_row, list):
            raise ValueError(
                f"Portrait map row {row_number} in {map_path} must be a list."
            )
        parsed_row: list[tuple[str, float]] = []
        for column_number, raw_cell in enumerate(raw_row, start=1):
            if not isinstance(raw_cell, (list, tuple)) or len(raw_cell) != 2:
                raise ValueError(
                    f"Portrait cell {row_number}:{column_number} in {map_path} "
                    "must be [character, opacity]."
                )
            character = str(raw_cell[0])
            if len(character) != 1:
                raise ValueError(
                    f"Portrait cell {row_number}:{column_number} in {map_path} "
                    "must contain exactly one character."
                )
            try:
                opacity = float(raw_cell[1])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid opacity at portrait cell {row_number}:{column_number}."
                ) from exc
            parsed_row.append((character, max(0.0, min(1.0, opacity))))
        portrait.append(parsed_row)

    return portrait


def render_portrait_line(
    row: list[tuple[str, float]],
    x: int,
    y: int,
) -> str:
    """Render one ASCII row while grouping adjacent equal-opacity cells."""
    if not row:
        return f'<text class="portrait-line" x="{x}" y="{y}"></text>'

    parts = [
        f'<text class="portrait-line" x="{x}" y="{y}" '
        'xml:space="preserve">'
    ]
    run_characters: list[str] = []
    run_opacity: float | None = None

    def flush() -> None:
        nonlocal run_characters, run_opacity
        if not run_characters or run_opacity is None:
            return
        run_text = escape("".join(run_characters))
        parts.append(
            f'<tspan fill-opacity="{run_opacity:.2f}">{run_text}</tspan>'
        )
        run_characters = []

    for character, opacity in row:
        quantized = round(opacity, 2)
        if run_opacity is None:
            run_opacity = quantized
        elif quantized != run_opacity:
            flush()
            run_opacity = quantized
        run_characters.append(character)

    flush()
    parts.append("</text>")
    return "".join(parts)


def render_svg(
    config: dict[str, Any],
    summary: dict[str, Any],
    portrait_rows: list[list[tuple[str, float]]],
    theme: Theme,
) -> str:
    layout = config.get("layout") or {}
    width = int(layout.get("width", 1100))
    height = int(layout.get("height", 580))
    font_size = int(layout.get("font_size", 16))
    portrait_font_size = int(layout.get("portrait_font_size", font_size))
    portrait_x = int(layout.get("portrait_x", 20))
    portrait_y = int(layout.get("portrait_y", 30))
    portrait_line_height = int(layout.get("portrait_line_height", 21))
    panel_x = int(layout.get("panel_x", 430))
    panel_columns = int(layout.get("panel_columns", 66))
    row_line_height = int(layout.get("row_line_height", 23))
    socials_heading_y = int(layout.get("socials_heading_y", 550))
    socials_badge_y = int(layout.get("socials_badge_y", 565))
    socials_badge_height = int(layout.get("socials_badge_height", 30))
    socials_gap = int(layout.get("socials_gap", 7))
    socials_font_size = int(layout.get("socials_font_size", 12))

    terminal_user = str(config.get("terminal_user", "user"))
    terminal_host = str(config.get("terminal_host", "host"))
    header = f"{terminal_user}@{terminal_host}"

    fields = {
        "username": str(summary["username"]),
        "account_age": str(summary["account_age"]),
        "age": personal_age(config),
        "display_name": str(config.get("display_name", "")),
    }

    portrait_markup = []
    for index, row in enumerate(portrait_rows):
        portrait_markup.append(
            render_portrait_line(
                row,
                x=portrait_x,
                y=portrait_y + index * portrait_line_height,
            )
        )

    panel_markup: list[str] = [render_header(panel_x, 31, panel_columns, header)]
    y = 55
    for row in config.get("rows") or []:
        if row.get("blank"):
            y += row_line_height
            continue
        label = apply_fields(row.get("label", ""), fields)
        value = apply_fields(row.get("value", ""), fields)
        panel_markup.append(render_regular_row(panel_x, y, panel_columns, label, value))
        y += row_line_height

    contact_heading_y = y + row_line_height
    panel_markup.append(
        render_header(panel_x, contact_heading_y, panel_columns, "- Contact")
    )
    y = contact_heading_y + row_line_height
    for row in config.get("contact_rows") or []:
        label = apply_fields(row.get("label", ""), fields)
        value = apply_fields(row.get("value", ""), fields)
        panel_markup.append(render_regular_row(panel_x, y, panel_columns, label, value))
        y += row_line_height

    stats_heading_y = y + row_line_height
    panel_markup.append(
        render_header(panel_x, stats_heading_y, panel_columns, "- GitHub Stats")
    )
    panel_markup.append(
        render_stats_line_one(
            panel_x, stats_heading_y + row_line_height, panel_columns, summary
        )
    )
    panel_markup.append(
        render_stats_line_two(
            panel_x, stats_heading_y + 2 * row_line_height, panel_columns, summary
        )
    )
    panel_markup.append(
        render_loc_line(
            panel_x, stats_heading_y + 3 * row_line_height, panel_columns, summary
        )
    )

    socials = configured_socials(config)
    panel_markup.append(
        render_header(panel_x, socials_heading_y, panel_columns, "- Socials")
    )
    social_max_width = max(1, width - panel_x - 34)
    social_badges_markup = render_social_badges(
        panel_x + 14,
        socials_badge_y,
        social_max_width - 14,
        socials_badge_height,
        socials_font_size,
        socials_gap,
        socials,
    )

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title description">
  <title id="title">{escape(str(config.get('alt_text', 'GitHub profile card')))}</title>
  <desc id="description">Terminal-style profile card for {escape(str(config.get('display_name', '')))} with an ASCII portrait and GitHub statistics.</desc>
  <defs>
    <filter id="shadow" x="-10%" y="-10%" width="120%" height="120%">
      <feDropShadow dx="0" dy="4" stdDeviation="8" flood-color="{theme.shadow}" flood-opacity="0.18"/>
    </filter>
  </defs>
  <style>
    text {{
      font-family: Consolas, "Liberation Mono", "DejaVu Sans Mono", "Courier New", monospace;
      font-variant-ligatures: none;
      font-feature-settings: "liga" 0;
      font-size: {font_size}px;
      fill: {theme.text};
      white-space: pre;
    }}
    .portrait-line {{ font-size: {portrait_font_size}px; }}
    .key {{ fill: {theme.key}; }}
    .value {{ fill: {theme.value}; }}
    .muted, .rule {{ fill: {theme.muted}; }}
    .addition {{ fill: {theme.addition}; }}
    .deletion {{ fill: {theme.deletion}; }}
    .social-badge-text {{
      fill: #ffffff;
      font-size: {socials_font_size}px;
      font-weight: 700;
      letter-spacing: 0.15px;
    }}
    .social-badge-icon {{ font-weight: 800; }}
  </style>
  <rect x="8" y="8" width="{width - 16}" height="{height - 16}" rx="18" fill="{theme.background}" stroke="{theme.border}" stroke-width="2" filter="url(#shadow)"/>
  <g aria-hidden="true">
    {''.join(portrait_markup)}
  </g>
  <text xml:space="preserve">
    {''.join(panel_markup)}
  </text>
  {social_badges_markup}
</svg>
"""


def main() -> int:
    args = parse_args()
    config_path = (ROOT / args.config).resolve()
    config = load_json(config_path, None)
    if not isinstance(config, dict):
        raise ValueError(f"Profile config must be a JSON object: {config_path}")

    username = resolve_username(config, args.username)
    summary, cache_path, request_count = collect_stats(
        config,
        username,
        offline=args.offline,
        skip_loc=args.skip_loc,
    )
    summary = normalize_summary(summary, username)

    portrait_path = ROOT / "portrait.txt"
    configured_map = str(config.get("portrait_map_file", "portrait_map.json")).strip()
    portrait_map_path = ROOT / configured_map if configured_map else None
    portrait_rows = load_portrait(portrait_path, portrait_map_path)

    dark_svg = render_svg(config, summary, portrait_rows, THEMES["dark"])
    light_svg = render_svg(config, summary, portrait_rows, THEMES["light"])
    (ROOT / "dark_mode.svg").write_text(dark_svg, encoding="utf-8")
    (ROOT / "light_mode.svg").write_text(light_svg, encoding="utf-8")
    update_readme_socials(config)

    print(f"Generated dark_mode.svg and light_mode.svg for @{summary['username']}.")
    print(f"Statistics cache: {cache_path.relative_to(ROOT)}")
    if request_count:
        print(f"GitHub GraphQL requests: {request_count}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2)
