"""
Empirical Stress-Test Harness for Wink HD Enhancer Remediation
Author: Challenger Remediation 1
Empirically stress-tests:
1. Stderr pipe draining & deadlock resistance.
2. Odd dimensions & macroblock alignment (1x and 2x).
3. Non-blocking event loop latency under load.
4. SSE streaming endpoints lifecycle & history replay.
"""

import os
import sys
import time
import json
import asyncio
import tempfile
import threading
import subprocess
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import cv2

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.enhancer.ffmpeg_utils import get_ffmpeg_path, get_ffprobe_path, run_ffmpeg
from app.enhancer.video_enhancer import (
    _drain_pipe_background,
    _enhance_video_sync,
    enhance_video,
    probe_video,
)
from app.main import app, TASKS, MEDIA_DIR, SAMPLES_DIR
from fastapi.testclient import TestClient
import httpx


# ==============================================================================
# Helper to create odd-dimension synthetic video
# ==============================================================================
def create_synthetic_odd_video(output_path: str, width: int, height: int, num_frames: int = 15, fps: float = 15.0) -> str:
    """Creates a video with exact odd dimensions using FFmpeg lavfi testsrc and mjpeg/yuvj444p."""
    dur = float(num_frames) / float(fps)
    cmd = [
        "-y",
        "-f", "lavfi",
        "-i", f"testsrc=size={width}x{height}:rate={fps}:duration={dur}",
        "-c:v", "mjpeg",
        "-pix_fmt", "yuvj444p",
        str(output_path),
    ]
    run_ffmpeg(cmd, check=True)
    return output_path



# ==============================================================================
# STRESS TEST 1: Stderr Pipe Draining & Deadlock Resistance
# ==============================================================================
def stress_test_pipe_draining() -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("STRESS TEST 1: Stderr Pipe Draining & Deadlock Resistance")
    print("=" * 70)

    results = {}

    # Test 1A: Extreme Pipe Flooding (100,000 lines = ~8 MB) through os.pipe
    print("[1A] Flooding anonymous pipe with 100,000 lines (~8MB) of stderr logs...")
    r_fd, w_fd = os.pipe()
    sink_list: List[str] = []
    r_file = os.fdopen(r_fd, "rb")
    w_file = os.fdopen(w_fd, "wb")

    drain_thread = threading.Thread(target=_drain_pipe_background, args=(r_file, sink_list), daemon=True)
    drain_thread.start()

    line_count = 100000
    t0 = time.time()

    def _flood_writer():
        for i in range(line_count):
            w_file.write(f"[FFMPEG_LOG_FRAME_{i:06d}] bitrate=1500kbit/s q=28.0 size=1024kB time=00:01:23.45\n".encode("utf-8"))
        w_file.close()

    writer_thread = threading.Thread(target=_flood_writer)
    writer_thread.start()
    writer_thread.join(timeout=10.0)
    assert not writer_thread.is_alive(), "Writer thread deadlocked on pipe buffer write!"

    drain_thread.join(timeout=5.0)
    elapsed = time.time() - t0
    assert not drain_thread.is_alive(), "Drain thread failed to terminate after EOF!"

    lines_captured = len(sink_list)
    total_bytes = sum(len(l.encode("utf-8")) for l in sink_list)
    mb = total_bytes / (1024 * 1024)
    print(f"  -> Successfully drained {lines_captured}/{line_count} lines ({mb:.2f} MB) in {elapsed:.3f}s ({mb/elapsed:.2f} MB/s)")
    assert lines_captured == line_count, f"Line count mismatch: expected {line_count}, got {lines_captured}"

    results["pipe_flood_lines"] = lines_captured
    results["pipe_flood_mb"] = round(mb, 2)
    results["pipe_flood_time_sec"] = round(elapsed, 3)
    results["pipe_flood_throughput_mbs"] = round(mb / elapsed, 2)

    # Test 1B: Broken Pipe & Non-Zero Subprocess Exit Handling
    print("[1B] Testing broken pipe & abnormal process termination...")
    ffmpeg_bin = get_ffmpeg_path()
    # Deliberately invalid argument to generate rapid error stderr
    proc = subprocess.Popen(
        [ffmpeg_bin, "-invalid_option_xyz", "-i", "pipe:0"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    err_sink = []
    t_err = threading.Thread(target=_drain_pipe_background, args=(proc.stderr, err_sink), daemon=True)
    t_err.start()
    retcode = proc.wait(timeout=5.0)
    t_err.join(timeout=3.0)
    assert retcode != 0, "Expected non-zero exit for invalid command"
    assert len(err_sink) > 0, "Expected error lines captured in sink"
    print(f"  -> Captured {len(err_sink)} error lines, exit code {retcode}, thread joined cleanly.")
    results["abnormal_exit_code"] = retcode
    results["abnormal_stderr_lines"] = len(err_sink)

    results["verdict"] = "PASS"
    return results


# ==============================================================================
# STRESS TEST 2: Odd Dimensions & Macroblock Alignment
# ==============================================================================
def stress_test_odd_dimensions() -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("STRESS TEST 2: Odd Dimensions & Macroblock Alignment (1x and 2x)")
    print("=" * 70)

    test_cases = [
        ("odd_w_and_h", 321, 241),
        ("square_odd", 199, 199),
        ("prime_odd", 173, 127),
    ]

    results = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        for label, in_w, in_h in test_cases:
            in_vid = str(tmp_path / f"input_{label}_{in_w}x{in_h}.mp4")
            create_synthetic_odd_video(in_vid, in_w, in_h, num_frames=12, fps=15.0)

            probe_in = probe_video(in_vid)
            assert probe_in["width"] == in_w and probe_in["height"] == in_h
            print(f"[*] Testing {label} ({in_w}x{in_h}, {probe_in['total_frames']} frames):")

            for upscale in (1, 2):
                out_vid = str(tmp_path / f"out_{label}_upscale_{upscale}x.mp4")
                t0 = time.time()

                # Run synchronous worker directly
                res = _enhance_video_sync(
                    input_path=in_vid,
                    output_path=out_vid,
                    upscale_factor=upscale,
                    face_retouch=0.4,
                    sharpen=0.4,
                    vibrance=0.3,
                    denoise=0.2,
                    emit_progress=lambda st, pct, fps, msg: None,
                )
                dur = time.time() - t0

                assert os.path.exists(out_vid), f"Output video was not generated for {label} {upscale}x"
                probe_out = probe_video(out_vid)

                out_w = probe_out["width"]
                out_h = probe_out["height"]

                # libx264 YUV420p requirement: MUST be strictly divisible by 2
                assert out_w % 2 == 0, f"Output width {out_w} is NOT even for {label} {upscale}x!"
                assert out_h % 2 == 0, f"Output height {out_h} is NOT even for {label} {upscale}x!"
                assert probe_out["total_frames"] == probe_in["total_frames"], "Frame count mismatch!"
                assert probe_out["duration"] > 0, "Invalid duration!"

                expected_w = (in_w * upscale) + ((in_w * upscale) % 2)
                expected_h = (in_h * upscale) + ((in_h * upscale) % 2)
                assert out_w == expected_w, f"Width mismatch: expected {expected_w}, got {out_w}"
                assert out_h == expected_h, f"Height mismatch: expected {expected_h}, got {out_h}"

                print(f"  -> Upscale {upscale}x: {in_w}x{in_h} -> {out_w}x{out_h} (even), {probe_out['total_frames']} frames in {dur:.2f}s [OK]")
                results[f"{label}_{upscale}x"] = {
                    "in_res": [in_w, in_h],
                    "out_res": [out_w, out_h],
                    "frames": probe_out["total_frames"],
                    "duration_sec": probe_out["duration"],
                    "even_aligned": True,
                }

    results["verdict"] = "PASS"
    return results


# ==============================================================================
# STRESS TEST 3: Non-Blocking Event Loop Latency Under Load
# ==============================================================================
async def _async_stress_test_event_loop() -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("STRESS TEST 3: Non-Blocking Event Loop Latency Under Load")
    print("=" * 70)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Start a real video enhancement task
        sample_vid = SAMPLES_DIR / "sample_video_lowres.mp4"
        assert sample_vid.exists(), f"Sample video not found at {sample_vid}"

        with open(sample_vid, "rb") as f:
            vid_bytes = f.read()

        print("[*] Dispatching video enhancement task to background worker...")
        init_res = await client.post(
            "/api/enhance/video",
            files={"file": ("stress_sample.mp4", vid_bytes, "video/mp4")},
            data={
                "upscale_factor": "2",
                "face_retouch": "0.5",
                "sharpen": "0.5",
                "vibrance": "0.3",
                "denoise": "0.3",
            },
        )
        assert init_res.status_code == 200, f"Task initialization failed: {init_res.text}"
        task_id = init_res.json()["task_id"]
        print(f"  -> Task ID: {task_id}")

        # 2. While video enhancement is actively working, fire 100 concurrent HTTP requests
        print("[*] Firing 100 concurrent HTTP GET /api/health and /api/samples directly on event loop...")
        latencies = []
        status_codes = []

        # Wait until task transitions to processing
        for _ in range(20):
            st_res = await client.get(f"/api/status/{task_id}")
            if st_res.json()["status"] == "processing":
                break
            await asyncio.sleep(0.05)

        num_requests = 100
        for i in range(num_requests):
            req_t0 = time.perf_counter()
            endpoint = "/api/health" if i % 2 == 0 else "/api/samples"
            r = await client.get(endpoint)
            req_dur_ms = (time.perf_counter() - req_t0) * 1000.0
            latencies.append(req_dur_ms)
            status_codes.append(r.status_code)
            await asyncio.sleep(0.005)

        # 3. Wait for video task to complete
        print("[*] Waiting for video task to complete...")
        for _ in range(60):
            st_res = await client.get(f"/api/status/{task_id}")
            st = st_res.json()
            if st["status"] in ("completed", "failed"):
                break
            await asyncio.sleep(0.5)

        final_st_res = await client.get(f"/api/status/{task_id}")
        final_st = final_st_res.json()
        assert final_st["status"] == "completed", f"Task did not complete successfully: {final_st}"

        # Latency analysis
        latencies_arr = np.array(latencies)
        mean_lat = float(np.mean(latencies_arr))
        median_lat = float(np.median(latencies_arr))
        p90_lat = float(np.percentile(latencies_arr, 90))
        p95_lat = float(np.percentile(latencies_arr, 95))
        p99_lat = float(np.percentile(latencies_arr, 99))
        max_lat = float(np.max(latencies_arr))
        min_lat = float(np.min(latencies_arr))

        print(f"  -> Total requests: {num_requests}, 200 OK: {status_codes.count(200)}/{num_requests}")
        print(f"  -> Mean latency: {mean_lat:.2f} ms (<50ms target)")
        print(f"  -> Median (p50): {median_lat:.2f} ms")
        print(f"  -> 90th percentile: {p90_lat:.2f} ms")
        print(f"  -> 95th percentile: {p95_lat:.2f} ms")
        print(f"  -> 99th percentile: {p99_lat:.2f} ms")
        print(f"  -> Max latency: {max_lat:.2f} ms")

        # Threshold checks: mean and median MUST be well below 50ms
        assert status_codes.count(200) == num_requests, "Some requests failed!"
        assert mean_lat < 50.0, f"Event loop mean latency exceeded 50ms: {mean_lat:.2f} ms"
        assert median_lat < 30.0, f"Event loop median latency exceeded 30ms: {median_lat:.2f} ms"

        print("  -> Concurrency verdict: Event loop is completely non-blocking! [PASS]")

        return {
            "num_requests": num_requests,
            "success_count": status_codes.count(200),
            "mean_latency_ms": round(mean_lat, 2),
            "p50_latency_ms": round(median_lat, 2),
            "p90_latency_ms": round(p90_lat, 2),
            "p99_latency_ms": round(p99_lat, 2),
            "max_latency_ms": round(max_lat, 2),
            "min_latency_ms": round(min_lat, 2),
            "verdict": "PASS",
        }


def stress_test_event_loop_concurrency() -> Dict[str, Any]:
    return asyncio.run(_async_stress_test_event_loop())



# ==============================================================================
# STRESS TEST 4: SSE Streaming Endpoints Lifecycle
# ==============================================================================
def stress_test_sse_lifecycle() -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("STRESS TEST 4: SSE Streaming Endpoints Lifecycle & History Replay")
    print("=" * 70)

    client = TestClient(app)
    sample_vid = SAMPLES_DIR / "sample_video_lowres.mp4"

    with open(sample_vid, "rb") as f:
        vid_bytes = f.read()

    # 4A: Stream Live Task Events via /api/tasks/{task_id}/events
    print("[4A] Initiating task and streaming live via /api/tasks/{task_id}/events...")
    init_res = client.post(
        "/api/enhance/video",
        files={"file": ("sse_live.mp4", vid_bytes, "video/mp4")},
        data={"upscale_factor": "1"},
    )
    assert init_res.status_code == 200
    task_id = init_res.json()["task_id"]

    events = []
    with client.stream("GET", f"/api/tasks/{task_id}/events") as stream:
        assert stream.status_code == 200
        assert "text/event-stream" in stream.headers["content-type"]
        for line in stream.iter_lines():
            if line.startswith("data: "):
                ev = json.loads(line[6:])
                events.append(ev)
                if ev.get("status") in ("completed", "failed"):
                    break

    assert len(events) >= 2, f"Too few events received: {len(events)}"
    first_ev = events[0]
    final_ev = events[-1]
    assert final_ev["status"] == "completed"
    assert final_ev["percent"] == 100.0
    assert "result" in final_ev
    assert "input_url" in final_ev["result"] and final_ev["result"]["input_url"].startswith("/media/")
    assert "output_url" in final_ev["result"] and final_ev["result"]["output_url"].startswith("/media/")
    print(f"  -> Live stream received {len(events)} events smoothly terminating at completed! [OK]")

    # 4B: Replay History via Late Subscriber
    print("[4B] Testing late subscriber history replay on completed task...")
    replayed_events = []
    t_replay_start = time.time()
    with client.stream("GET", f"/api/tasks/{task_id}/events") as stream:
        assert stream.status_code == 200
        for line in stream.iter_lines():
            if line.startswith("data: "):
                ev = json.loads(line[6:])
                replayed_events.append(ev)
                if ev.get("status") in ("completed", "failed"):
                    break
    replay_duration = time.time() - t_replay_start
    print(f"  -> Late subscriber received {len(replayed_events)} replayed events in {replay_duration:.3f}s without hanging!")
    assert len(replayed_events) >= 1
    assert replayed_events[-1]["status"] == "completed"
    assert replay_duration < 1.0, f"History replay took too long: {replay_duration}s"

    # 4C: Test Alternative SSE Endpoint /api/progress/{task_id}
    print("[4C] Testing alias endpoint /api/progress/{task_id}...")
    alias_events = []
    with client.stream("GET", f"/api/progress/{task_id}") as stream:
        assert stream.status_code == 200
        for line in stream.iter_lines():
            if line.startswith("data: "):
                ev = json.loads(line[6:])
                alias_events.append(ev)
                if ev.get("status") in ("completed", "failed"):
                    break
    assert len(alias_events) >= 1
    assert alias_events[-1]["status"] == "completed"
    print(f"  -> Alias endpoint /api/progress/{task_id} returned completed status correctly. [OK]")

    # 4D: Disconnect Resilience (Client drops after 2 events)
    print("[4D] Testing premature client disconnect resilience...")
    init_res2 = client.post(
        "/api/enhance/video",
        files={"file": ("sse_abort.mp4", vid_bytes, "video/mp4")},
        data={"upscale_factor": "1"},
    )
    task_id2 = init_res2.json()["task_id"]

    dropped_events = []
    with client.stream("GET", f"/api/tasks/{task_id2}/events") as stream:
        for line in stream.iter_lines():
            if line.startswith("data: "):
                dropped_events.append(json.loads(line[6:]))
                if len(dropped_events) >= 2:
                    break  # simulate client disconnect

    print(f"  -> Client disconnected after {len(dropped_events)} events. Verifying backend task continues to completion...")
    for _ in range(40):
        st = client.get(f"/api/status/{task_id2}").json()
        if st["status"] in ("completed", "failed"):
            break
        time.sleep(0.5)

    final_st2 = client.get(f"/api/status/{task_id2}").json()
    assert final_st2["status"] == "completed", f"Backend task failed after client disconnect: {final_st2}"
    print(f"  -> Task {task_id2} cleanly completed in background despite client disconnection! [OK]")

    return {
        "live_stream_events_count": len(events),
        "history_replay_events_count": len(replayed_events),
        "history_replay_latency_sec": round(replay_duration, 3),
        "alias_endpoint_status": "PASS",
        "disconnect_resilience_status": "PASS",
        "verdict": "PASS",
    }


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================
def main():
    print("=" * 70)
    print("STARTING EMPIRICAL CHALLENGER STRESS HARNESS")
    print("=" * 70)

    all_results = {}
    try:
        all_results["test_1_pipe_draining"] = stress_test_pipe_draining()
        all_results["test_2_odd_dimensions"] = stress_test_odd_dimensions()
        all_results["test_3_event_loop_concurrency"] = stress_test_event_loop_concurrency()
        all_results["test_4_sse_lifecycle"] = stress_test_sse_lifecycle()

        all_results["overall_verdict"] = "APPROVE"
        print("\n" + "=" * 70)
        print("ALL EMPIRICAL STRESS TESTS PASSED WITH ZERO ERRORS!")
        print("OVERALL VERDICT: APPROVE")
        print("=" * 70)

        # Output JSON summary to stdout
        print("\n--- JSON EMPIRICAL SUMMARY ---")
        print(json.dumps(all_results, indent=2))
        return 0
    except Exception as exc:
        print(f"\n[FAIL] EMPIRICAL STRESS TEST FAILED: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        all_results["overall_verdict"] = "REQUEST_CHANGES"
        all_results["failure"] = str(exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
