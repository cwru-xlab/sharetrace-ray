from __future__ import annotations

from typing import Any, Hashable, Iterable, Union


class Actor:
    """An actor as defined in the actor-based model of computing.

    Attributes:
        name: A hashable value identifies the actor.
        mailbox: A buffer that stores messages received from other actors.
        neighbors: A mapping from actor names to their inboxes.
    """
    __slots__ = ('name', 'mailbox', 'neighbors')

    def __init__(self, name: Hashable, mailbox=None, **kwargs):
        self.name = name
        self.mailbox = mailbox
        self.neighbors = {}

    def on_next(self, msg, **kwargs) -> None:
        """Executes upon receiving a message."""
        pass

    def on_start(self, *args, **kwargs) -> None:
        """Executes upon initializing the actor."""
        pass

    def on_stop(self, *args, **kwargs) -> None:
        """Executes prior to terminating the actor."""
        pass

    def send(self, *msgs, **kwargs) -> None:
        """Sends messages to other actors."""
        pass

    def receive(self) -> Any:
        """Receives a message from other actors."""
        pass

    def connect(self, *actors: Actor, duplex: bool = False) -> None:
        """Enables this actor to send messages to other actors."""
        self.neighbors.update((a.name, a.mailbox) for a in actors)
        if duplex:
            for a in (a for a in actors if a is not self):
                a.connect(self)

    def disconnect(self, *actors: Actor) -> None:
        """Disables this actor from sending messages to other actors."""
        pop = self.neighbors.pop
        for a in actors:
            pop(a.name, None)

    def __repr__(self):
        return f'{self.__class__.__name__}(name={self.name})'

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, o: object) -> bool:
        return self is o or (isinstance(o, Actor) and self.name == o.name)


Actors = Union[Iterable, Actor]


class ActorSystem(Actor):
    """Coordinates a collection of actors.

        Attributes:
            actors: A collection of actors in the system.
    """

    __slots__ = ('actors',)

    def __init__(self, name: Hashable, mailbox=None, **kwargs):
        super().__init__(name, mailbox, **kwargs)
        self.actors = set()

    def connect(self, *actors: Actors, duplex: bool = False) -> None:
        """Connects actors to each other and the system."""
        add = self.actors.add
        for a in actors:
            if isinstance(a, Actor):
                super().connect(a, duplex=duplex)
                add(a)
            elif isinstance(a, Iterable) and len(a := list(a)) == 2:
                a1, a2 = a
                super().connect(a1, a2, duplex=duplex)
                a1.connect(a2, duplex=duplex)
                add(a1)
                add(a2)
            else:
                raise TypeError(
                    f'Input must be an Actor or a pair of Actors; got {a}')

    def __repr__(self):
        cls = self.__class__.__name__
        return f'{cls}(name={self.name}, actors={len(self.actors)})'
