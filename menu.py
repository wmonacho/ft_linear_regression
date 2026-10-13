"""Interactive launcher: pick between training, prediction and reset from a menu."""

import itertools
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from linear_regression.model import THETAS_FILE, load_thetas

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"
CLEAR_LINE = "\033[2K\r"

BANNER = r"""
  __ _      _ _                                                    _
 / _| |_   | (_)_ __   ___  __ _ _ __   _ __ ___  __ _ _ __ ___  ___(_) ___  _ __
| |_| __|  | | | '_ \ / _ \/ _` | '__| | '__/ _ \/ _` | '__/ _ \/ __| |/ _ \| '_ \
|  _| |_   | | | | | |  __/ (_| | |    | | |  __/ (_| | | |  __/\__ \ | (_) | | | |
|_|  \__|  |_|_|_| |_|\___|\__,_|_|    |_|  \___|\__, |_|  \___||___/_|\___/|_| |_|
                                                 |___/
"""
GRADIENT = (51, 45, 39, 33, 27, 63, 99)

DEFAULT_DATASET = "data/data.csv"
MENU = (
    ("train", "Train the model"),
    ("predict", "Predict a price"),
    ("status", "Show current parameters"),
    ("reset", "Reset training (delete thetas.json)"),
    ("quit", "Quit"),
)


def main() -> None:
    try:
        _show_banner()
        while True:
            choice = _select(MENU)
            print()
            if choice == "quit":
                break
            ACTIONS[choice]()
            _pause()
    except (KeyboardInterrupt, EOFError):
        print()
    finally:
        print(SHOW_CURSOR, end="")
    _typewrite(f"{CYAN}Bye!{RESET}\n")


def _show_banner() -> None:
    """Reveal the banner line by line with a color gradient."""
    print("\033[2J\033[H", end="")
    for i, line in enumerate(BANNER.strip("\n").splitlines()):
        print(f"\033[38;5;{GRADIENT[i % len(GRADIENT)]}m{line}{RESET}")
        time.sleep(0.06)
    _typewrite(f"{DIM}   Predict the price of a car from its mileage.{RESET}\n\n")


def _typewrite(text: str, delay: float = 0.012) -> None:
    """Print text one character at a time, writing color codes instantly."""
    in_escape = False
    for char in text:
        sys.stdout.write(char)
        if char == "\033":
            in_escape = True
        elif in_escape:
            in_escape = char != "m"
        else:
            sys.stdout.flush()
            time.sleep(delay)
    sys.stdout.flush()


def _select(options: tuple[tuple[str, str], ...]) -> str:
    """Let the user pick an option with the arrow keys, or by number if not a terminal."""
    if not sys.stdin.isatty():
        return _select_by_number(options)

    import termios
    import tty

    index = 0
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    print(f"{BOLD}What do you want to do?{RESET} {DIM}(↑/↓ then Enter){RESET}")
    print(HIDE_CURSOR, end="")
    try:
        tty.setcbreak(fd)
        while True:
            _draw_menu(options, index)
            key = sys.stdin.read(1)
            if key == "\033":
                key += sys.stdin.read(2)
            if key in ("\033[A", "k"):
                index = (index - 1) % len(options)
            elif key in ("\033[B", "j"):
                index = (index + 1) % len(options)
            elif key.isdigit() and 1 <= int(key) <= len(options):
                index = int(key) - 1
                _draw_menu(options, index)
                return options[index][0]
            elif key in ("\n", "\r"):
                return options[index][0]
            elif key in ("q", "\x04"):
                return "quit"
            print(f"\033[{len(options)}A", end="")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        print(SHOW_CURSOR, end="")


def _draw_menu(options: tuple[tuple[str, str], ...], index: int) -> None:
    for i, (_, label) in enumerate(options):
        if i == index:
            print(f"{CLEAR_LINE}  {CYAN}{BOLD}❯ {label}{RESET}")
        else:
            print(f"{CLEAR_LINE}    {DIM}{label}{RESET}")
    sys.stdout.flush()


def _select_by_number(options: tuple[tuple[str, str], ...]) -> str:
    for i, (_, label) in enumerate(options, start=1):
        print(f"  {i}. {label}")
    while True:
        answer = input("Choice: ").strip()
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return options[int(answer) - 1][0]
        print(f"{RED}Please enter a number between 1 and {len(options)}.{RESET}")


def _train() -> None:
    path = input(f"Dataset {DIM}[{DEFAULT_DATASET}]{RESET}: ").strip() or DEFAULT_DATASET
    result = _with_spinner("Training with gradient descent", _run, "train.py", path)
    if result.returncode != 0:
        print(result.stderr or result.stdout, end="")
        return
    print(f"{GREEN}✓ Training complete.{RESET}")
    _status()


def _predict() -> None:
    print(f"{DIM}─── predict.py ───{RESET}", flush=True)
    # Ctrl-C must stop predict.py only, not the menu around it.
    previous_handler = signal.signal(signal.SIGINT, lambda *_: None)
    try:
        subprocess.run([sys.executable, "predict.py"])
    finally:
        signal.signal(signal.SIGINT, previous_handler)
    print(f"{DIM}──────────────────{RESET}")


def _status() -> None:
    try:
        theta0, theta1 = load_thetas()
    except ValueError as e:
        print(f"{RED}✗ {e}{RESET}")
        return
    if not Path(THETAS_FILE).exists():
        print(f"{YELLOW}⚠ Model not trained: θ0 = θ1 = 0.{RESET}")
        return
    _typewrite(f"  θ0 = {BOLD}{theta0:.4f}{RESET}\n")
    _typewrite(f"  θ1 = {BOLD}{theta1:.6f}{RESET}\n")
    _typewrite(f"  {DIM}price = {theta0:.2f} + ({theta1:.6f} × km){RESET}\n")


def _reset() -> None:
    if not Path(THETAS_FILE).exists():
        print(f"{YELLOW}⚠ Nothing to reset: the model is not trained.{RESET}")
        return
    answer = input(f"{YELLOW}Delete {THETAS_FILE}? [y/N]{RESET} ").strip().lower()
    if answer != "y":
        print(f"{DIM}Cancelled.{RESET}")
        return
    Path(THETAS_FILE).unlink()
    print(f"{GREEN}✓ Training reset: θ0 = θ1 = 0.{RESET}")


def _run(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, script, *args], capture_output=True, text=True)


def _with_spinner(message: str, func, *args):
    """Run func in a thread while showing a spinner, for at least a short moment."""
    result = {}
    thread = threading.Thread(target=lambda: result.setdefault("value", func(*args)))
    thread.start()
    start = time.monotonic()
    print(HIDE_CURSOR, end="")
    for frame in itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"):
        if not thread.is_alive() and time.monotonic() - start > 0.8:
            break
        print(f"{CLEAR_LINE}{CYAN}{frame}{RESET} {message}...", end="", flush=True)
        time.sleep(0.08)
    print(CLEAR_LINE + SHOW_CURSOR, end="")
    thread.join()
    return result["value"]


def _pause() -> None:
    input(f"\n{DIM}Press Enter to go back to the menu...{RESET}")
    print()


ACTIONS = {
    "train": _train,
    "predict": _predict,
    "status": _status,
    "reset": _reset,
}


if __name__ == "__main__":
    main()
