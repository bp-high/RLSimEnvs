"""The line: points west to east (terminals, stations, halts, junctions) and the sections between them.

A section joins two neighbouring points and is single or double track. A point has a number of tracks (None: as many
as needed, the terminals), may be *directional* (a station on double track: one track per direction, so two trains
going the same way cannot both stand there and nothing overtakes), and may be a *turn* point where a train can be
short-turned. Junctions are where double track becomes single between two stations; trains pass them but never stop.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

_UP = Path(__file__).resolve().parents
DATA = (_UP[4] if len(_UP) > 4 else _UP[-1]) / "data"     # the repository's data/: only curation and builds read it


@dataclass
class Point:
    code: str
    name: str
    km: float
    kind: str              # terminal | station | halt | junction
    tracks: int | None     # None: unlimited (terminal)
    directional: bool      # one track per direction (station on double track)
    turn: bool             # trains may be short-turned here

    @property
    def stoppable(self) -> bool:
        return self.kind != "junction"


@dataclass
class Section:
    a: str                 # west end
    b: str                 # east end
    single: bool
    prorail: str = ""


class Network:
    def __init__(self, points: list[Point], sections: list[Section], line: str = "", name: str = ""):
        self.line, self.name = line, name
        self.points = points
        self.sections = sections
        self.index = {p.code: i for i, p in enumerate(points)}
        self.by_code = {p.code: p for p in points}
        for i, s in enumerate(sections):
            if (self.index[s.a], self.index[s.b]) != (i, i + 1):
                raise ValueError(f"section {s.a}-{s.b} is not between neighbouring points {i} and {i + 1}")

    @classmethod
    def from_dict(cls, d: dict) -> "Network":
        return cls([Point(**p) for p in d["points"]], [Section(**s) for s in d["sections"]], d.get("line", ""),
                   d.get("name", ""))

    def to_dict(self) -> dict:
        return {"line": self.line, "name": self.name, "points": [asdict(p) for p in self.points],
                "sections": [asdict(s) for s in self.sections]}

    def section(self, x: str, y: str) -> int:
        """Index of the section between neighbouring points x and y (either order)."""
        i, j = self.index[x], self.index[y]
        if abs(i - j) != 1:
            raise ValueError(f"{x} and {y} are not neighbours")
        return min(i, j)

    def route(self, a: str, b: str) -> list[str]:
        """Every point from a to b inclusive, in travel order."""
        i, j = self.index[a], self.index[b]
        step = 1 if j >= i else -1
        return [self.points[k].code for k in range(i, j + step, step)]

    def direction(self, a: str, b: str) -> int:
        """+1 eastbound (towards the last point), -1 westbound."""
        return 1 if self.index[b] > self.index[a] else -1


def load_network(line: str = "gnlw") -> Network:
    return Network.from_dict(json.loads((DATA / line / "network.json").read_text()))
