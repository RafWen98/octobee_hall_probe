"""The searchable index behind the Help tab."""



from octobee import help as ohelp
from tests.helpers import (
    check,
)



def test_help_index():
    """The Help tab is only as good as its search."""
    print("\nhelp index")
    topics = ohelp.load_topics()
    n_gui = sum(1 for t in topics if t.source == "this window")
    check("the README is indexed into topics", len(topics) > 20,
          f"{len(topics)} topics, {n_gui} of them about the window")
    # The window's own topics exist to cover what a document about the
    # instrument cannot. If they start multiplying, they are being written
    # instead of the README, which is how help text and docs drift apart.
    check("the window's own topics stay a small minority", n_gui < 8,
          f"{n_gui} hand-written topics against {len(topics) - n_gui} indexed")
    check("every topic has a title and a body",
          all(t.title and t.body.strip() for t in topics))

    # Headings inside fenced code are comments, not topics. This one bit:
    # the README is full of bash blocks whose lines start with '#'.
    fenced = ohelp.split_markdown(
        "## Real\ntext\n\n```bash\n## not a heading\necho hi\n```\n\n"
        "## Also real\nmore\n")
    check("a '#' inside a code fence is not a heading",
          [t.title for t in fenced] == ["Real", "Also real"],
          str([t.title for t in fenced]))

    for query, want in (("homing", "Homing, and why the window asks"),
                        ("jog step loud", "Why a bigger jog step is louder, "
                                          "and what to do about it"),
                        ("guided magnet", "Step 5b"),
                        # Typed as a question, which is how anyone actually
                        # reaches for help. Without stop-word filtering this
                        # ranks by document length instead of by relevance.
                        ("why is it so loud", "Everything got slow, or loud"),
                        ("go is greyed out", "Go is greyed out, or a move is "
                                             "refused")):
        hits = ohelp.search(topics, query, limit=3)
        check(f"searching {query!r} finds its topic first",
              bool(hits) and hits[0].title.startswith(want),
              hits[0].title if hits else "nothing matched")
    check("a query of nothing but stop words still answers",
          bool(ohelp.search(topics, "how do I use the")),
          "dropping every term would leave a blank pane")
    check("a query that matches nothing returns nothing",
          ohelp.search(topics, "zzzqqq") == [])
    check("an empty query lists everything",
          len(ohelp.search(topics, "   ", limit=500)) == len(topics))
