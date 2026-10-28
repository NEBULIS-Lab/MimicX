"""Public setup examples must be syntactically usable without private tools."""
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]


def test_install_shell_examples_parse_without_executing_setup():
    text = (ROOT / "docs/ISAAC_INSTALL.md").read_text()
    blocks = [part.split("```", 1)[0] for part in text.split("```bash\n")[1:]]
    assert len(blocks) >= 7
    for block in blocks:
        subprocess.run(["bash", "-n"], input=block, text=True, check=True, capture_output=True)
        if block.startswith("python3 - <<'PY'\n"):
            compile(block.split("\n", 1)[1].rsplit("\nPY", 1)[0], "documented source receipt", "exec")


def test_install_uses_public_pins_and_records_source_identity_without_container_git():
    text = (ROOT / "docs/ISAAC_INSTALL.md").read_text()
    pins = json.loads((ROOT / "dependencies/isaaclab-sources.json").read_text())
    assert pins["isaac_lab"]["commit"] and pins["whole_body_tracking"]["commit"]
    for required in ("isaaclab-sources.json", "isaaclab.txt", "checkout --detach", "sha256sum --check",
                     "--cpu --shell", "--system-site-packages", "--no-deps --no-build-isolation",
                     "GIT_PYTHON_REFRESH=quiet", "shutil.which(\"git\") is None",
                     "source-identities.json", "ACCEPT_EULA=Y", "PRIVACY_CONSENT=Y"):
        assert required in text
    for private in ("/data/private/", "/data/shared/", "install-cpu.sh", "/runtime/tools/bin/git"):
        assert private not in text
    assert "(ISAAC_INSTALL.md)" in (ROOT / "docs/ISAAC.md").read_text()
