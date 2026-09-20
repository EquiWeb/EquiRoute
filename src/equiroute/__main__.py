"""Module entry point for EquiRoute."""


def main() -> None:
    from .cli import app

    app()


if __name__ == "__main__":
    main()
