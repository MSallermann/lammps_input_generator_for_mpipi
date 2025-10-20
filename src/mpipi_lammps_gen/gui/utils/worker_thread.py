import threading


class PlayPauseThread(threading.Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pause_event = threading.Event()
        self._pause_event.set()  # start unpaused
        self._stop_event = threading.Event()

    def run(self):
        try:
            if self._target is not None:
                while not self._stop_event.is_set():
                    self._pause_event.wait()  # block here if paused
                    self._target(*self._args, **self._kwargs)
        finally:
            # Avoid a refcycle if the thread is running a function with
            # an argument that has a member that points to the thread.
            del self._target, self._args, self._kwargs

    def pause(self):
        self._pause_event.clear()  # pause

    def resume(self):
        self._pause_event.set()  # resume

    def stop(self):
        self._stop_event.set()
        self._pause_event.set()  # in case it's paused


if __name__ == "__main__":
    import time

    thread = PlayPauseThread(target=lambda: print("a"))
    thread.start()
    time.sleep(0.0001)
    thread.pause()
    print("paused")
    time.sleep(5)
    thread.resume()
    time.sleep(0.0001)
    thread.stop()
