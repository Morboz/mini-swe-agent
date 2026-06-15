#!/usr/bin/env python3

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import typer


app = typer.Typer(add_completion=False)


@dataclass
class PullFailure:
    instance_id: str
    image: str
    attempts: int
    waited_seconds: int
    returncode: int
    error: str


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_image_name(instance_id: str, registry: str) -> str:
    docker_instance_id = instance_id.replace("__", "_1776_")
    return f"{registry.rstrip('/')}/sweb.eval.x86_64.{docker_instance_id}:latest".lower()


def append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def append_text(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def pull_image(
    docker_bin: str,
    image: str,
    max_retry_seconds: int,
    base_delay_seconds: int,
) -> PullFailure | None:
    attempts = 0
    waited_seconds = 0
    while True:
        attempts += 1
        log(f"attempt {attempts}: pulling {image}")
        completed = subprocess.run([docker_bin, "pull", image], capture_output=True, text=True, check=False)
        if completed.returncode == 0:
            return None

        error = (completed.stderr or completed.stdout).strip()
        remaining = max_retry_seconds - waited_seconds
        if remaining <= 0:
            return PullFailure(
                instance_id="",
                image=image,
                attempts=attempts,
                waited_seconds=waited_seconds,
                returncode=completed.returncode,
                error=error,
            )

        delay = min(base_delay_seconds * (2 ** (attempts - 1)), remaining)
        log(f"pull failed for {image}, exit={completed.returncode}, retry in {delay}s")
        waited_seconds += delay
        time.sleep(delay)


@app.command()
def main(
    dataset: Path = typer.Option(Path("django_cases.jsonl"), help="Django case JSONL dataset."),
    registry: str = typer.Option("docker.1panel.live/swebench", help="Registry prefix."),
    docker_bin: str = typer.Option("docker", help="Docker executable to use."),
    pulled_file: Path = typer.Option(Path("django_pulled_images.txt"), help="File containing pulled images."),
    failed_file: Path = typer.Option(Path("django_failed_images.jsonl"), help="File containing failed images."),
    max_retry_seconds: int = typer.Option(300, help="Max retry budget per image in seconds."),
    base_delay_seconds: int = typer.Option(10, help="Base delay for exponential backoff."),
    limit: int | None = typer.Option(None, help="Optional limit for testing or partial pulls."),
) -> None:
    records = read_jsonl(dataset)
    if limit is not None:
        records = records[:limit]

    pulled_file.write_text("", encoding="utf-8")
    failed_file.write_text("", encoding="utf-8")

    total = len(records)
    log(f"dataset: {dataset}")
    log(f"cases to pull: {total}")
    log(f"pulled output: {pulled_file}")
    log(f"failed output: {failed_file}")

    pulled_count = 0
    failed_count = 0
    for index, record in enumerate(records, start=1):
        instance_id = record["instance_id"]
        image = build_image_name(instance_id, registry)
        log(f"[{index}/{total}] start {instance_id}")
        failure = pull_image(
            docker_bin=docker_bin,
            image=image,
            max_retry_seconds=max_retry_seconds,
            base_delay_seconds=base_delay_seconds,
        )
        if failure is None:
            pulled_count += 1
            append_text(pulled_file, image)
            log(f"[{index}/{total}] success {instance_id}")
            continue

        failed_count += 1
        failure.instance_id = instance_id
        append_jsonl(
            failed_file,
            {
                "instance_id": failure.instance_id,
                "image": failure.image,
                "attempts": failure.attempts,
                "waited_seconds": failure.waited_seconds,
                "returncode": failure.returncode,
                "error": failure.error,
            },
        )
        log(f"[{index}/{total}] failed {instance_id}")

    log(f"finished, pulled={pulled_count}, failed={failed_count}")
    log(f"pulled file: {pulled_file}")
    log(f"failed file: {failed_file}")


if __name__ == "__main__":
    app()