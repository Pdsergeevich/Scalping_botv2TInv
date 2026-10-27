"""Канал событий с единственным читателем (аналог ``System.Threading.Channels``).

Живой режим использует ``asyncio.Queue`` (неблокирующая запись при unbounded),
бэктест — тот же объект, но события читаются синхронно через ``try_read``.
"""

from __future__ import annotations

import asyncio
from typing import Generic, Iterable, List, Optional, TypeVar

T = TypeVar("T")


class Channel(Generic[T]):
    """Очередь «много писателей / один читатель»."""

    __slots__ = ("_queue", "_maxsize", "_closed")

    def __init__(self, maxsize: int = 0) -> None:
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._maxsize = maxsize
        self._closed = False

    def __len__(self) -> int:
        return self._queue.qsize()

    @property
    def closed(self) -> bool:
        return self._closed

    async def write(self, item: T) -> None:
        if self._closed:
            return
        await self._queue.put(item)

    def try_write(self, item: T) -> bool:
        """Запись без ожидания. ``False`` — канал заполнен."""
        if self._closed:
            return False
        if self._maxsize and self._queue.qsize() >= self._maxsize:
            return False
        self._queue.put_nowait(item)
        return True

    async def read(self) -> Optional[T]:
        """Читает элемент; ``None`` — канал закрыт и пуст."""
        if self._closed and self._queue.empty():
            return None
        return await self._queue.get()

    def try_read(self) -> Optional[T]:
        if self._queue.empty():
            return None
        return self._queue.get_nowait()

    def drain(self) -> List[T]:
        items: List[T] = []
        while True:
            item = self.try_read()
            if item is None:
                return items
            items.append(item)

    def extend(self, items: Iterable[T]) -> None:
        for item in items:
            self._queue.put_nowait(item)

    def close(self) -> None:
        self._closed = True
