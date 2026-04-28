#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import httpx
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled
from youtube_transcript_api._transcripts import FetchedTranscript, FetchedTranscriptSnippet


def load_netscape_cookies(cookies_file: str) -> dict[str, str]:
    """Load YouTube-only cookies from a Netscape-format cookies.txt file (exported by yt-dlp).
    Filters to youtube.com domain only to avoid 413 Request Entity Too Large errors."""
    cookies: dict[str, str] = {}
    with open(cookies_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 7:
                domain, name, value = parts[0], parts[5], parts[6]
                if "youtube.com" in domain:
                    cookies[name] = value
    return cookies


def make_api(cookies_file: str | None) -> YouTubeTranscriptApi:
    """Create a YouTubeTranscriptApi instance, optionally using browser cookies."""
    if cookies_file:
        cookies = load_netscape_cookies(cookies_file)
        client = httpx.Client(cookies=cookies, follow_redirects=True)
        return YouTubeTranscriptApi(http_client=client)
    return YouTubeTranscriptApi()


MANIFEST_FIELDS = [
    "video_id",
    "title",
    "video_url",
    "status",
    "language",
    "is_generated",
    "json_path",
    "text_path",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bulk download YouTube transcripts with timestamps."
    )
    parser.add_argument(
        "channel_url",
        help="YouTube channel URL, handle URL, playlist URL, or /videos URL.",
    )
    parser.add_argument(
        "--output-dir",
        default="transcripts",
        help="Directory where transcript files will be written.",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["en"],
        help="Preferred transcript languages in priority order.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N videos.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.5,
        help="Delay between transcript requests.",
    )
    parser.add_argument(
        "--include-generated",
        action="store_true",
        help="Keep auto-generated transcripts. By default they are skipped.",
    )
    parser.add_argument(
        "--write-text",
        action="store_true",
        help="Also write a readable text transcript alongside JSON.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip videos already marked ok or skipped_generated in an existing manifest.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of retries for transient transcript fetch failures.",
    )
    parser.add_argument(
        "--skip-title",
        action="append",
        default=[],
        help="Skip videos whose titles contain this text. Repeat for multiple filters.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip videos when the expected JSON transcript file already exists.",
    )
    parser.add_argument(
        "--cookies",
        default=None,
        metavar="COOKIES_FILE",
        help=(
            "Path to a Netscape-format cookies.txt file (bypasses IP blocks). "
            "Generate with: yt-dlp --cookies-from-browser chrome --cookies youtube_cookies.txt https://www.youtube.com"
        ),
    )
    return parser.parse_args()


def normalize_channel_url(url: str) -> str:
    url = url.strip().rstrip("/")
    if "youtube.com" not in url and "youtu.be" not in url:
        raise ValueError("Expected a YouTube URL.")
    if "/videos" in url or "/playlist" in url or "list=" in url:
        return url
    return f"{url}/videos"


def slugify(value: str, max_length: int = 120) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return slug[:max_length] or "untitled"


def format_timestamp(seconds: float) -> str:
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


@dataclass
class VideoEntry:
    video_id: str
    title: str
    webpage_url: str


def load_channel_videos(channel_url: str, limit: int | None) -> list[VideoEntry]:
    yt_dlp_command = ["yt-dlp"] if shutil.which("yt-dlp") else [sys.executable, "-m", "yt_dlp"]
    yt_dlp_args = [
        "--flat-playlist",
        "--dump-single-json",
    ]
    if limit:
        yt_dlp_args.extend(["--playlist-end", str(limit)])
    command = [*yt_dlp_command, *yt_dlp_args, channel_url]

    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "yt-dlp is not installed. Install it first with `pip install -r requirements.txt`."
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() or exc.stdout.strip()
        raise RuntimeError(f"yt-dlp failed to load the channel: {stderr}") from exc

    payload = json.loads(result.stdout)
    entries = payload.get("entries") or []
    videos: list[VideoEntry] = []

    for item in entries:
        video_id = item.get("id")
        if not video_id:
            continue
        title = item.get("title") or video_id
        webpage_url = item.get("url") or f"https://www.youtube.com/watch?v={video_id}"
        if webpage_url.startswith("/"):
            webpage_url = f"https://www.youtube.com{webpage_url}"
        if "watch?v=" not in webpage_url:
            webpage_url = f"https://www.youtube.com/watch?v={video_id}"
        videos.append(VideoEntry(video_id=video_id, title=title, webpage_url=webpage_url))

    return videos


def transcript_rows(transcript: Iterable[FetchedTranscriptSnippet]) -> list[dict]:
    rows: list[dict] = []
    for item in transcript:
        rows.append(
            {
                "text": item.text.replace("\n", " ").strip(),
                "start": float(item.start),
                "duration": float(item.duration),
                "timestamp": format_timestamp(float(item.start)),
            }
        )
    return rows


def fetch_transcript(
    api: YouTubeTranscriptApi,
    video_id: str,
    languages: list[str],
) -> FetchedTranscript:
    return api.fetch(video_id, languages=languages, preserve_formatting=True)


def load_previous_results(manifest_path: Path) -> dict[str, dict[str, str]]:
    if not manifest_path.exists():
        return {}
    with manifest_path.open("r", newline="", encoding="utf-8") as manifest_file:
        return {row["video_id"]: row for row in csv.DictReader(manifest_file)}


def fetch_with_retries(
    api: YouTubeTranscriptApi,
    video_id: str,
    languages: list[str],
    retries: int,
    base_sleep_seconds: float,
) -> FetchedTranscript:
    for attempt in range(retries + 1):
        try:
            return fetch_transcript(api, video_id, languages)
        except (NoTranscriptFound, TranscriptsDisabled):
            raise
        except Exception:
            if attempt >= retries:
                raise
            time.sleep(max(base_sleep_seconds, 1.0) * (2 ** attempt))
    raise RuntimeError("Unreachable")


def write_json(output_path: Path, payload: dict) -> None:
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def write_text(output_path: Path, title: str, video_url: str, rows: list[dict]) -> None:
    lines = [title, video_url, ""]
    for row in rows:
        lines.append(f"[{row['timestamp']}] {row['text']}")
    output_path.write_text("\n".join(lines) + "\n")


def should_skip_video(title: str, skip_patterns: list[str]) -> str | None:
    normalized_title = title.casefold()
    for pattern in skip_patterns:
        if pattern.casefold() in normalized_title:
            return pattern
    return None


def is_blocked_error(error: str) -> bool:
    lowered = error.casefold()
    return "youtube is blocking requests from your ip" in lowered or "requestblocked" in lowered or "ipblocked" in lowered


def write_manifest(manifest_path: Path, rows: list[dict[str, str]]) -> None:
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        channel_url = normalize_channel_url(args.channel_url)
        videos = load_channel_videos(channel_url, args.limit)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    api = make_api(args.cookies)
    manifest_path = output_dir / "manifest.csv"
    previous_results = load_previous_results(manifest_path) if args.resume else {}
    rows_to_write: list[dict[str, str]] = []

    if previous_results:
        for video_id, row in previous_results.items():
            if row.get("status") in {"ok", "skipped_generated", "skipped_filter"}:
                rows_to_write.append(row)

    order_by_video_id = {video.video_id: index for index, video in enumerate(videos, start=1)}

    for index, video in enumerate(videos, start=1):
        if previous_results.get(video.video_id, {}).get("status") in {"ok", "skipped_generated", "skipped_filter"}:
            print(f"[{index}/{len(videos)}] Skipping already processed: {video.title}")
            continue

        safe_name = f"{index:04d}-{slugify(video.title)}"
        json_path = output_dir / f"{safe_name}.json"
        text_path = output_dir / f"{safe_name}.txt"
        matched_skip_pattern = should_skip_video(video.title, args.skip_title)

        if args.skip_existing and json_path.exists():
            print(f"[{index}/{len(videos)}] skipped_existing: {video.title}")
            rows_to_write.append(
                {
                    "video_id": video.video_id,
                    "title": video.title,
                    "video_url": video.webpage_url,
                    "status": "skipped_existing",
                    "language": "",
                    "is_generated": "",
                    "json_path": str(json_path),
                    "text_path": str(text_path) if text_path.exists() else "",
                    "error": "Skipped because transcript JSON already exists.",
                }
            )
            write_manifest(manifest_path, sorted(rows_to_write, key=lambda row: order_by_video_id.get(row["video_id"], 10**9)))
            continue

        if matched_skip_pattern:
            print(f"[{index}/{len(videos)}] skipped_filter: {video.title}")
            rows_to_write.append(
                {
                    "video_id": video.video_id,
                    "title": video.title,
                    "video_url": video.webpage_url,
                    "status": "skipped_filter",
                    "language": "",
                    "is_generated": "",
                    "json_path": "",
                    "text_path": "",
                    "error": f"Skipped because title matched filter: {matched_skip_pattern}",
                }
            )
            write_manifest(manifest_path, sorted(rows_to_write, key=lambda row: order_by_video_id.get(row["video_id"], 10**9)))
            continue

        print(f"[{index}/{len(videos)}] fetching: {video.title}")
        try:
            transcript = fetch_with_retries(
                api,
                video.video_id,
                args.languages,
                retries=args.retries,
                base_sleep_seconds=args.sleep_seconds,
            )
            rows = transcript_rows(transcript)
            metadata = {
                "video_id": video.video_id,
                "title": video.title,
                "video_url": video.webpage_url,
                "language": transcript.language,
                "language_code": transcript.language_code,
                "is_generated": transcript.is_generated,
                "segments": rows,
            }

            if metadata["is_generated"] and not args.include_generated:
                rows_to_write.append(
                    {
                        "video_id": video.video_id,
                        "title": video.title,
                        "video_url": video.webpage_url,
                        "status": "skipped_generated",
                        "language": metadata["language_code"],
                        "is_generated": str(metadata["is_generated"]),
                        "json_path": "",
                        "text_path": "",
                        "error": "Transcript is auto-generated. Re-run with --include-generated to keep it.",
                    }
                )
                print(f"[{index}/{len(videos)}] skipped_generated: {video.title}")
                write_manifest(manifest_path, sorted(rows_to_write, key=lambda row: order_by_video_id.get(row["video_id"], 10**9)))
                continue

            write_json(json_path, metadata)
            if args.write_text:
                write_text(text_path, video.title, video.webpage_url, rows)

            rows_to_write.append(
                {
                    "video_id": video.video_id,
                    "title": video.title,
                    "video_url": video.webpage_url,
                    "status": "ok",
                    "language": metadata["language_code"],
                    "is_generated": str(metadata["is_generated"]),
                    "json_path": str(json_path),
                    "text_path": str(text_path) if args.write_text else "",
                    "error": "",
                }
            )
            print(f"[{index}/{len(videos)}] ok: {video.title}")
        except (NoTranscriptFound, TranscriptsDisabled) as exc:
            rows_to_write.append(
                {
                    "video_id": video.video_id,
                    "title": video.title,
                    "video_url": video.webpage_url,
                    "status": "missing",
                    "language": "",
                    "is_generated": "",
                    "json_path": "",
                    "text_path": "",
                    "error": str(exc),
                }
            )
            print(f"[{index}/{len(videos)}] missing: {video.title}")
        except Exception as exc:
            error_text = str(exc)
            status = "blocked" if is_blocked_error(error_text) else "error"
            rows_to_write.append(
                {
                    "video_id": video.video_id,
                    "title": video.title,
                    "video_url": video.webpage_url,
                    "status": status,
                    "language": "",
                    "is_generated": "",
                    "json_path": "",
                    "text_path": "",
                    "error": error_text,
                }
            )
            print(f"[{index}/{len(videos)}] {status}: {video.title}")

        write_manifest(manifest_path, sorted(rows_to_write, key=lambda row: order_by_video_id.get(row["video_id"], 10**9)))

        if args.sleep_seconds:
            time.sleep(args.sleep_seconds)

    rows_to_write.sort(key=lambda row: order_by_video_id.get(row["video_id"], 10**9))
    write_manifest(manifest_path, rows_to_write)

    print(f"Wrote manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
