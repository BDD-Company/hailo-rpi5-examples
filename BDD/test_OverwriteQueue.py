# test_overwrite_queue.py
import threading
import time
import queue
import pytest

from OverwriteQueue import OverwriteQueue

def test_overwrite_drops_oldest():
    q = OverwriteQueue(maxlen=3)
    q.put(1)
    q.put(2)
    q.put(3)
    q.put(4)  # overwrite oldest (1)

    assert len(q) == 3
    assert q.get_nowait() == 2
    assert q.get_nowait() == 3
    assert q.get_nowait() == 4
    with pytest.raises(queue.Empty):
        q.get_nowait()


def test_get_nowait_empty_raises():
    q = OverwriteQueue(maxlen=2)
    with pytest.raises(queue.Empty):
        q.get_nowait()


def test_get_blocks_until_item_available():
    q = OverwriteQueue(maxlen=2)
    out = {}

    def consumer():
        out["value"] = q.get()

    t = threading.Thread(target=consumer, daemon=True)
    t.start()

    time.sleep(0.05)
    assert "value" not in out  # still blocked

    q.put("hello")
    t.join(timeout=1.0)

    assert out["value"] == "hello"


def test_get_timeout_raises_queue_empty():
    q = OverwriteQueue(maxlen=2)

    t0 = time.perf_counter()
    with pytest.raises(queue.Empty):
        q.get(timeout=0.05)
    dt = time.perf_counter() - t0

    # avoid flaky CI timing
    assert dt >= 0.03


def test_unblocks_single_waiter():
    q = OverwriteQueue(maxlen=1)
    got = []

    def consumer():
        got.append(q.get())

    t = threading.Thread(target=consumer, daemon=True)
    t.start()

    time.sleep(0.02)
    q.put(123)

    t.join(timeout=1.0)
    assert got == [123]


def test_fifo_when_not_overwritten():
    q = OverwriteQueue(maxlen=128)
    N = 10_000
    out = []

    start = threading.Barrier(2)

    def producer():
        start.wait()
        for i in range(N):
            q.put(i)
            # slow down the producer and allow consumer to pick up
            # 50 is arbitrary discovered by experimenting
            if i % 50 == 0: time.sleep(0)

    def consumer():
        start.wait()
        for _ in range(N):
            out.append(q.get(timeout=1.0))

    tp = threading.Thread(target=producer, daemon=True)
    tc = threading.Thread(target=consumer, daemon=True)

    tp.start()
    tc.start()

    tp.join(timeout=2.0)
    tc.join(timeout=2.0)

    assert len(out) == N
    assert out[:100] == list(range(100))
    assert out[-1] == N - 1



def test_overwrite_keeps_latest_items():
    q = OverwriteQueue(maxlen=50)
    N = 20_000
    done = threading.Event()
    seen = []

    def producer():
        for i in range(N):
            q.put(i)
        done.set()

    def consumer():
        while True:
            try:
                seen.append(q.get(timeout=0.05))
            except queue.Empty:
                if done.is_set() and len(q) == 0:
                    return

    tp = threading.Thread(target=producer, daemon=True)
    tc = threading.Thread(target=consumer, daemon=True)

    tc.start()
    tp.start()
    tp.join()
    tc.join()

    assert seen
    assert seen[-1] == N - 1
    assert all(a < b for a, b in zip(seen[-100:-1], seen[-99:]))


if __name__ == "__main__":
    import pytest
    import sys

    if len(sys.argv) == 1:
        sys.exit(pytest.main(["-qq", __file__]))
    else:
        sys.exit(pytest.main())