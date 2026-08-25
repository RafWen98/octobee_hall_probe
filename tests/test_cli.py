"""The command-line front door."""

import importlib


from octobee.cli import main as cli
from tests.helpers import (
    check,
)



def test_cli_commands():
    """Every subcommand the front door advertises must actually be there.

    The dispatcher names modules as strings, so a module that moves or a main()
    that is renamed breaks a command that nothing else exercises -- and breaks
    it at the moment someone types it, which is the worst time to find out.
    """
    print("\ncommand line")
    check("the usage listing names every command",
          all(name in cli._usage() for name in cli.COMMANDS),
          str(sorted(cli.COMMANDS)))
    missing = []
    for name, (modname, _) in sorted(cli.COMMANDS.items()):
        try:
            mod = importlib.import_module(modname)
        except Exception as exc:
            missing.append(f"{name}: {modname} will not import ({exc})")
            continue
        if not callable(getattr(mod, "main", None)):
            missing.append(f"{name}: {modname} has no main()")
    check(f"all {len(cli.COMMANDS)} subcommands resolve to a main()",
          not missing, "; ".join(missing))
    check("an unknown command is refused, not run",
          cli.main(["definitely-not-a-command"]) == 2)
