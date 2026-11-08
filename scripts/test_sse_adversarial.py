"""
Adversarial Challenge Verification for SSE Event Replay & Concurrency
Author: Challenger Iteration 2
"""
import sys
import time
import json
import threading
from unittest.mock import patch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from app.main import app, TASKS, SAMPLES_DIR

def test_sse_concurrent_and_late_joiners():
    print("\n--- ADVERSARIAL TEST 1: Concurrent & Staggered SSE Joiners ---")
    client = TestClient(app)
    sample_vid = SAMPLES_DIR / "sample_video_lowres.mp4"
    with open(sample_vid, "rb") as f:
        vid_bytes = f.read()

    # 1. Start video enhancement
    res = client.post(
        "/api/enhance/video",
        files={"file": ("adv_test.mp4", vid_bytes, "video/mp4")},
        data={"upscale_factor": "1"},
    )
    assert res.status_code == 200
    task_id = res.json()["task_id"]

    results = {
        "early_client": [],
        "mid_client": [],
        "late_client": [],
        "post_complete_client": [],
    }

    # Early client connects immediately
    def run_early_client():
        with client.stream("GET", f"/api/tasks/{task_id}/events") as stream:
            for line in stream.iter_lines():
                if line.startswith("data: "):
                    ev = json.loads(line[6:])
                    results["early_client"].append(ev)
                    if ev.get("status") in ("completed", "failed"):
                        break

    # Mid client connects after 0.5s delay
    def run_mid_client():
        time.sleep(0.5)
        with client.stream("GET", f"/api/tasks/{task_id}/events") as stream:
            for line in stream.iter_lines():
                if line.startswith("data: "):
                    ev = json.loads(line[6:])
                    results["mid_client"].append(ev)
                    if ev.get("status") in ("completed", "failed"):
                        break

    t_early = threading.Thread(target=run_early_client)
    t_mid = threading.Thread(target=run_mid_client)

    t_early.start()
    t_mid.start()

    t_early.join(timeout=15.0)
    t_mid.join(timeout=15.0)

    # Late client connects after early client finishes
    t0 = time.time()
    with client.stream("GET", f"/api/progress/{task_id}") as stream:
        for line in stream.iter_lines():
            if line.startswith("data: "):
                ev = json.loads(line[6:])
                results["late_client"].append(ev)
                if ev.get("status") in ("completed", "failed"):
                    break
    late_duration = time.time() - t0

    # Another client connects well after completion
    time.sleep(0.1)
    with client.stream("GET", f"/api/tasks/{task_id}/events") as stream:
        for line in stream.iter_lines():
            if line.startswith("data: "):
                ev = json.loads(line[6:])
                results["post_complete_client"].append(ev)
                if ev.get("status") in ("completed", "failed"):
                    break

    print(f"Early client received: {len(results['early_client'])} events")
    print(f"Mid client received: {len(results['mid_client'])} events")
    print(f"Late client received: {len(results['late_client'])} events in {late_duration:.4f}s")
    print(f"Post-complete client received: {len(results['post_complete_client'])} events")

    # Assertions
    assert len(results["early_client"]) >= 5, "Early client received fewer than 5 events!"
    assert len(results["late_client"]) == len(results["early_client"]), "Late client did not receive identical history!"
    assert len(results["post_complete_client"]) == len(results["early_client"]), "Post-complete client count mismatch!"
    assert late_duration < 0.1, f"Late client replay took {late_duration}s (expected <0.1s)!"

    # Verify chronological monotonic progression of percentages in replayed stream
    percentages = [e["percent"] for e in results["late_client"]]
    print(f"Progression of percentages in replayed stream: {percentages}")
    assert percentages == sorted(percentages), "Percentages are not monotonically non-decreasing!"
    assert percentages[-1] == 100.0, "Terminal percentage must be 100.0!"

    print("[PASS] ALL ADVERSARIAL CONCURRENCY AND REPLAY CHECKS PASSED!")


def test_sse_background_failure_streaming():
    print("\n--- ADVERSARIAL TEST 2: Background Failure SSE Event Streaming ---")
    client = TestClient(app)
    sample_vid = SAMPLES_DIR / "sample_video_lowres.mp4"
    with open(sample_vid, "rb") as f:
        vid_bytes = f.read()

    # We mock enhance_video to raise an exception during execution
    async def mock_failing_enhance(*args, **kwargs):
        progress_cb = kwargs.get("progress_callback")
        if progress_cb:
            await progress_cb("probing", 5.0, 0.0, "Started probing...")
        raise RuntimeError("Simulated background pipeline crash")

    with patch("app.main.enhance_video", side_effect=mock_failing_enhance):
        res = client.post(
            "/api/enhance/video",
            files={"file": ("failing_vid.mp4", vid_bytes, "video/mp4")},
            data={"upscale_factor": "1"},
        )
        assert res.status_code == 200
        task_id = res.json()["task_id"]

        events = []
        t0 = time.time()
        with client.stream("GET", f"/api/tasks/{task_id}/events") as stream:
            for line in stream.iter_lines():
                if line.startswith("data: "):
                    ev = json.loads(line[6:])
                    events.append(ev)
                    if ev.get("status") in ("completed", "failed", "error"):
                        break
        dur = time.time() - t0

        print(f"Failing task stream received {len(events)} events in {dur:.3f}s")
        assert len(events) >= 2, "Expected intermediate probing event + failure event"
        final_ev = events[-1]
        assert final_ev["status"] == "failed"
        assert "Simulated background pipeline crash" in final_ev["error"]
        print(f"Captured failure event: {final_ev['message']}")

        # Late subscriber check on failed task
        t_late = time.time()
        replayed = []
        with client.stream("GET", f"/api/tasks/{task_id}/events") as stream:
            for line in stream.iter_lines():
                if line.startswith("data: "):
                    ev = json.loads(line[6:])
                    replayed.append(ev)
                    if ev.get("status") in ("completed", "failed", "error"):
                        break
        late_dur = time.time() - t_late
        print(f"Late subscriber on failed task received {len(replayed)} events in {late_dur:.4f}s")
        assert len(replayed) == len(events)
        assert replayed[-1]["status"] == "failed"
        assert late_dur < 0.1

    print("[PASS] ALL ADVERSARIAL FAILURE STREAMING CHECKS PASSED!")


if __name__ == "__main__":
    test_sse_concurrent_and_late_joiners()
    test_sse_background_failure_streaming()
    print("\n" + "=" * 50)
    print("ALL ADVERSARIAL CHALLENGE TESTS PASSED CLEANLY!")
    print("=" * 50)
