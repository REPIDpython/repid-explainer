"""Console script for repid_explainer."""

import click


@click.command()
def main():
    """Main entrypoint."""
    click.echo("repid-explainer")
    click.echo("=" * len("repid-explainer"))
    click.echo("REPID: Regional Effect Plots with implicit Interaction Detection")


if __name__ == "__main__":
    main()  # pragma: no cover
