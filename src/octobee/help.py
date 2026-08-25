#!/usr/bin/env python3
"""
octobee/help.py -- the searchable index behind the GUI's Help tab.

Where the text comes from
-------------------------
README.md, split at its headings, plus a handful of topics about the window
itself that have no place in a document about the instrument.

Indexing the README rather than writing separate help text is the whole point.
This project's README is where the reasoning already lives -- why the roll
sweep needs three bearings, why a scan stops at every point, why a bigger jog
step is louder -- and a second copy of that in a help pane would be wrong
within a month. Nobody proof-reads help text against a README. So there is one
source, and the window reads it.

The consequence is that the help is only as good as the README, which is the
right incentive: a topic that is hard to find here is a heading that needs
rewriting there.

Searching
---------
Plain substring matching over title and body, ranked. Not fuzzy, not stemmed,
no index built ahead of time: the whole corpus is about 75 kB, so scoring every
section on every keystroke costs well under a millisecond, and anything
cleverer would need explaining to whoever maintains it. Title hits outrank body
hits, whole-word hits outrank partial ones, and a section that contains all the
words outranks one that contains any.
"""

import os
import re

from octobee import paths

README_NAME = "README.md"

# Only these levels become topics. `#` is the document title and `####` and
# below are usually a detail inside a topic rather than one of their own.
HEADING_RE = re.compile(r"^(#{2,3})\s+(.*?)\s*#*$", re.M)

# Fenced code must not be scanned for headings: a '## ' inside a bash block is
# a comment, and splitting on it produces a topic made of half a command.
FENCE_RE = re.compile(r"^```", re.M)


class Topic:
    """One help entry: a title, the text under it, and where it came from."""

    def __init__(self, title, body, source="README.md", level=2):
        self.title = title
        self.body = body
        self.source = source
        self.level = level
        self._hay = (title + "\n" + body).lower()

    def __repr__(self):
        return f"Topic({self.title!r}, {len(self.body)} chars)"

    def score(self, terms):
        """How well this topic answers a query. 0 means it does not."""
        if not terms:
            return 0.0
        title = self.title.lower()
        total = 0.0
        hit = 0
        for t in terms:
            n_title = title.count(t)
            n_body = self._hay.count(t) - n_title
            if not (n_title or n_body):
                continue
            hit += 1
            # A word in the heading is the strongest signal there is: headings
            # in this README are written as the question being answered.
            total += 12.0 * n_title + min(n_body, 8)
            if re.search(rf"\b{re.escape(t)}\b", title):
                total += 8.0
        if not hit:
            return 0.0
        # Every word present beats some words present, whatever the counts:
        # "jog step loud" should not be beaten by a section that says "step"
        # forty times and never mentions jogging.
        return total * (1.0 + hit / len(terms)) * (2.0 if hit == len(terms) else 1.0)


def split_markdown(text, source=README_NAME):
    """Markdown -> [Topic], one per ## or ### heading, code fences respected."""
    # Blank out fenced blocks for the purposes of FINDING headings, while
    # keeping the real text for the bodies.
    masked = list(text)
    inside = False
    for m in FENCE_RE.finditer(text):
        inside = not inside
        if inside:
            start = m.start()
        else:
            for i in range(start, m.end()):
                if masked[i] == "#":
                    masked[i] = " "
    masked = "".join(masked)

    topics = []
    marks = list(HEADING_RE.finditer(masked))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        body = text[m.end():end].strip("\n")
        topics.append(Topic(m.group(2).strip(), body, source=source,
                            level=len(m.group(1))))

    # A heading whose next line is another heading has no text of its own --
    # it is a section title. Selecting one in a list and getting a blank pane
    # reads as a broken index, so it becomes a contents page for what it
    # covers, which is the only honest thing it has to say.
    for i, t in enumerate(topics):
        if t.body.strip():
            continue
        under = []
        for other in topics[i + 1:]:
            if other.level <= t.level:
                break
            under.append(other)
        t.body = ("This section covers:\n\n"
                  + "\n".join(f"* {o.title}" for o in under)
                  if under else "*(this heading has no text of its own)*")
        t._hay = (t.title + "\n" + t.body).lower()
    return topics


# --------------------------------------------------------------------------
# topics about the window, which the README has no reason to carry
# --------------------------------------------------------------------------
#
# Deliberately few, and deliberately symptom-shaped. Everything here that could
# be written as an explanation of the INSTRUMENT belongs in the README instead
# and is indexed from there -- these are the entries someone reaches for when
# the window is not doing what they expected, plus the tour you cannot get from
# a document about hardware. If a topic here starts restating a README section,
# delete it: two copies is how help text goes stale.

GUI_TOPICS = [
    ("Getting started: what each tab is for",
     """
**Live** — the rolling plot, the sensor bar chart and the 3D probe head. Start
here; if the field is wrong, it is wrong here first. Dragging or scrolling
either plot stops it auto-ranging, with nothing on screen to say so; **reset
view** on each of them puts the axes back on auto.

**Sensors** — the 16 sensors as numbers, with per-channel health. The place to
look when one sensor disagrees with the rest.

**Calibration** — in order: the zero point, a magnet pass to equalise
response, the Earth-field roll sweep that matches the sensors properly, and the
calibration file itself.

**Diagnostics** — railed, stuck and noisy channels, and the report you send to
someone else.

**Stages** — the three translators: axis map, jogging, homing, motion profile,
and the field map.

**Data output** — recording and one-shot exports.

**Profile** — where the frame time goes, when the window feels slow.

**Log** — everything the window has done this session, with timings.

The window connects itself when it opens — carriers and stages together. Start
it with `--no-connect` if you would rather it did not.
""".strip()),

    ("Guided magnet calibration: the checklist",
     """
Before starting the wizard on the Calibration tab:

1. **Clamp the magnet** beside the head's travel, level with one face and a few
   centimetres clear of it. It must not move again until all four poses are
   done — it is the common reference for every sensor.
2. **Home the axes**, or at least the sweep axis.
3. **Park the head** so the magnet is just clear of the first ring of sensors.
   The sweep starts from wherever the head is now.
4. Nothing ferrous may move nearby during a sweep, including you.

Then, four times: the wizard drives the sweep and stops to ask you to index the
tube one quarter turn **about its own axis and nothing else**. A pose that also
shifts the head sideways breaks the routine's whole argument quietly — the
numbers still look plausible afterwards.

The exact angle does not matter. Four faces each getting their turn does, and
so does the head being rigid between poses: same cradle, same cable dress.

*Why this beats a hand pass, and what it measures: see the README topic*
**Step 5b — the guided magnet run**.
""".strip()),

    ("Go is greyed out, or a move is refused",
     """
Almost always one of these:

* **The emergency stop is latched.** The button in the top right reads
  **STOPPED** and a **Reset** button has appeared beside it. Nothing moves
  until that is reset, on purpose. The Log says what tripped it.
* **The stages are not connected.** Press Connect — it opens the carriers and
  the stages together — or connect them from the Stages tab. Without an axis
  map in `stages.json` there is nothing to connect to; assign x/y/z once on the
  Stages tab and every later session picks it up.
* **The position cannot be trusted.** Absolute moves need a reference. That is
  usually "not homed yet", but an axis can also *lose* its reference without
  the controller admitting it — after an emergency stop, a stall or a
  collision, the steps it counted are no longer the millimetres it moved. The
  state column then reads **POSITION LOST** rather than **NOT HOMED**, and the
  cure is the same: home it. Jogging is relative and works either way.
* **Something else is already driving.** A field map or a guided magnet run
  owns the axes for its duration, and the jog, Go and Home buttons go dead
  while it does — two threads commanding the same axis is how a map ends up
  full of points taken somewhere other than where it says.
* **The target is outside the working envelope.** `limit_mm` in `stages.json`
  is what the axis is *allowed* to use, and it is normally smaller than the
  300 mm of travel, because the fixture and the head are inside that travel.

A field map additionally refuses before it starts if its range leaves the
envelope, or if any axis it would scan has no trustworthy position — a map
with no origin is worse than no map, because it looks entirely plausible.
""".strip()),

    ("The emergency stop, and what it is not",
     """
**The red button, top right of the window. Escape does the same thing.** It is
there on every tab, and it works whether or not the stages are connected.

Pressing it does three things:

1. **Stops every axis immediately.** Immediate, not profiled — the
   deceleration ramp is abandoned. That is the point of it: at the default
   6 mm/s and 10 mm/s² a *profiled* stop still coasts about 1.8 mm, and if
   1.8 mm did not matter you would not be reaching for the button.
2. **Latches all motion off.** Every axis, every thread, until you reset it.
   Stopping only the axis that happens to be moving does nothing about the
   raster thread that is a fraction of a second away from commanding the next
   point.
3. **Marks the positions untrustworthy.** An immediate stop can lose steps, so
   afterwards the count no longer matches the carriage — even though the
   controller still reports the axis as homed. Absolute moves stay refused
   until you home it again. **Reset clears the latch; it does not tell the
   machine where the head is.**

**Stop moving**, on the Stages tab, is the graded version: profiled, does not
latch, nothing needs re-homing. Use that for "that is far enough" and the red
one for "something is wrong".

### What it is not

This is a software stop, sent over USB, by this program. It needs this program
running, the USB link up and the controllers answering. The failures a real
emergency stop exists for are exactly the ones where those are not true — the
PC wedged, the cable out, the process killed. In those the motors carry on
doing whatever they were last told.

**If this rig can hurt someone, the stop that matters is a mushroom head wired
in series with the controllers' supply.** Nothing in software substitutes for
it.

If the red button reports that it could not reach an axis, cut power to the
controllers.

### From another terminal

    python octobee/motion/stage.py estop

for when the window is not the thing that is running, or is not answering.
That stops the axes but cannot latch anything — the latch lives in a process,
and that one exits — so stop whatever commanded the move as well.
""".strip()),

    ("Everything got slow, or loud",
     """
**Loud** is the motion profile. Kinesis ships these stages at 20 mm/s and
20 mm/s², which puts any move past about a 5 mm step into the steppers'
resonance band. The default here is 6 mm/s and 10 mm/s², set on the Stages tab
beside the jog buttons, with a hard ceiling of 10 mm/s that nothing can exceed.
See the README topic *Why a bigger jog step is louder*.

**Slow to move** is the same setting seen from the other side: capping the
speed costs about 22 s on a full 300 mm traverse. Raising the *acceleration*
while leaving the speed cap alone usually gets the time back without the peak
speed going up.

**Slow to draw** is the Profile tab's business. The 3D probe head is the most
expensive thing in the window; untick **3D** on the Live tab and nothing else
changes — acquisition, calibration and recording are unaffected.
""".strip()),
]


def gui_topics():
    return [Topic(t, body.strip(), source="this window", level=2)
            for t, body in GUI_TOPICS]


def load_topics(readme_path=None):
    """Every topic the Help tab offers: the window's own, then the README's.

    A missing README is not an error -- the window still explains itself, and
    says where the rest went.
    """
    topics = gui_topics()
    root = paths.repo_root()
    path = readme_path or (os.path.join(root, README_NAME) if root else README_NAME)
    try:
        with open(path, encoding="utf-8") as fh:
            topics += split_markdown(fh.read(), source=os.path.basename(path))
    except OSError:
        topics.append(Topic(
            "The README could not be read",
            f"Help topics are indexed from `{path}`, which is not readable "
            f"from here. Only the topics about this window are available.",
            source="this window"))
    return topics


# People type questions, not keywords -- "what does the sensors tab do", "why
# is it so loud". These words appear in every section of a 75 kB document, so
# scoring them rewards length instead of relevance. Dropped only when doing so
# leaves something behind, or searching for "how it works" would find nothing.
STOPWORDS = frozenset((
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do",
    "does", "for", "from", "get", "how", "i", "if", "in", "is", "it", "its",
    "me", "my", "not", "of", "on", "or", "so", "than", "that", "the", "then",
    "there", "this", "to", "use", "used", "using", "was", "what", "when",
    "where", "which", "why", "will", "with", "you", "your"))


def search(topics, query, limit=40):
    """Topics matching `query`, best first. An empty query returns them all."""
    terms = [t for t in re.split(r"\W+", query.lower()) if t]
    strong = [t for t in terms if t not in STOPWORDS]
    terms = strong or terms
    if not terms:
        return list(topics)[:limit]
    scored = [(t.score(terms), i, t) for i, t in enumerate(topics)]
    hits = sorted(((s, i, t) for s, i, t in scored if s > 0),
                  key=lambda kv: (-kv[0], kv[1]))
    return [t for _s, _i, t in hits[:limit]]
