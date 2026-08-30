"""Opt-in stack dump support for the isolated remote smoke process."""

from __future__ import annotations

import faulthandler
import signal


faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)

