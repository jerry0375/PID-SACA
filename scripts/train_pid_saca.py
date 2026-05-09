"""Train the PID-SACA method from the repository root."""

import _bootstrap  # noqa: F401
from pid_saca.algorithms import pid_saca


def main() -> None:
    config = pid_saca.Config()
    pid_saca.train_saca(config)


if __name__ == "__main__":
    main()
