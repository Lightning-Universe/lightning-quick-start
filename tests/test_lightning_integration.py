from click.testing import CliRunner
from lightning.app.cli.lightning_cli import run_app


def test_lightning_can_use_external_component():
    runner = CliRunner()
    result = runner.invoke(
        run_app,
        [
            "tests/integration_app/app.py",
            "--blocking",
            "False",
            "--open-ui",
            "False",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
