#!/usr/bin/env python3
"""
Smart re-ingest script: cleans and re-vectorises all knowledge base content.

Changes applied:
  1. FAQ section (418 Q&A pairs) → single shared 'utel_faq' collection.
     Each Q&A pair = 1 complete chunk (no mid-answer splits).
  2. Each materia → academic content only (no duplicated FAQ).
     Chunks tagged with tipo='academico'.
  3. JSON sources tagged tipo='academico'.

Run inside the container:
    docker exec botacademia_api python scripts/reingest_smart.py
"""
import asyncio
import sys
import os
import time

# Ensure /app is on the path when running inside Docker
sys.path.insert(0, "/app")

from app.services.ingest_service import ingest_materia, ingest_faq
from app.core.logging import get_logger

logger = get_logger("reingest")

MATERIAS = [
    "152_Introduccion_a_la_administracion_publica_C",
    "155_Principios_y_perspectivas_de_la_administracion_C",
    "156_Sociologia_Rural_C",
    "258_Criminologia_B",
    "58_Estadistica_y_probabilidad",
]


async def main() -> None:
    total_start = time.perf_counter()

    print("\n" + "="*60)
    print("  BotAcademia Smart Re-Ingest")
    print("="*60)
    print(f"\n  Materias  : {len(MATERIAS)}")
    print("  Strategy  : Academic-only per materia + shared FAQ collection")
    print("  FAQ chunks: 1 chunk per Q&A pair (no mid-answer splits)")
    print("="*60 + "\n")

    # ── Step 1: Shared FAQ collection ────────────────────────────────────────
    print("[1/6] Ingesting shared FAQ → utel_faq ...")
    t = time.perf_counter()
    result = await ingest_faq(force_reingest=True)
    ms = round((time.perf_counter() - t) * 1000)
    status_icon = "✓" if result["status"] in ("indexed", "already_indexed") else "✗"
    print(f"  {status_icon} {result['status']} | {result['num_chunks']} Q&A chunks | {ms}ms")
    print(f"    {result['message']}\n")

    # ── Step 2-6: Each materia (academic content only) ────────────────────
    for step, materia_id in enumerate(MATERIAS, start=2):
        short = materia_id[:50]
        print(f"[{step}/{len(MATERIAS)+1}] {short} ...")
        t = time.perf_counter()
        result = await ingest_materia(materia_id, force_reingest=True)
        ms = round((time.perf_counter() - t) * 1000)
        status_icon = "✓" if result["status"] in ("indexed", "already_indexed") else "✗"
        print(f"  {status_icon} {result['status']} | {result['num_chunks']} chunks | {ms}ms\n")

    total_ms = round((time.perf_counter() - total_start) * 1000)
    print("="*60)
    print(f"  COMPLETE in {total_ms/1000:.1f}s")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
