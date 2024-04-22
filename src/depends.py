from queue import Queue


class MaxSizeQueue(Queue):
    def __init__(self, maxsize: int = 0) -> None:
        super().__init__(maxsize=maxsize)

    def put(self, item, block: bool = False, timeout: float | None = None) -> None:
        if self.full():
            self.get()
        super().put(item, block, timeout)

    def show(self):
        print(list(self.queue))
