"""Console script for ModelStudy_FR."""

import typer
from rich.console import Console

from ModelStudy_FR import utils

app = typer.Typer()
console = Console()


@app.command()
def main():
    """Console script for ModelStudy_FR."""
    console.print("Replace this message by putting your code into "
               "ModelStudy_FR.cli.main")
    console.print("See Typer documentation at https://typer.tiangolo.com/")
    utils.do_something_useful()


if __name__ == "__main__":
    app()
