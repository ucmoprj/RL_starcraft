"""Small runtime fixes for known PySC2 bugs.

You do not need to understand this file to follow the course.
It is imported automatically by the environments in this folder.

Fix 1 (Windows): PySC2 passes an `extra_ports` argument all the way down to
`subprocess.Popen`, which does not accept it, so the game fails to launch with
    TypeError: Popen.__init__() got an unexpected keyword argument 'extra_ports'
We simply drop that argument before the game process is started.
"""
from pysc2.lib import sc_process

_original_launch = sc_process.StarcraftProcess._launch


def _launch_without_extra_ports(self, run_config, args, **kwargs):
    kwargs.pop("extra_ports", None)
    return _original_launch(self, run_config, args, **kwargs)


if not getattr(sc_process.StarcraftProcess, "_course_patched", False):
    sc_process.StarcraftProcess._launch = _launch_without_extra_ports
    sc_process.StarcraftProcess._course_patched = True
