from pathlib import Path

from setuptools import find_packages, setup


def load_requirements() -> list[str]:
    requirements_path = Path(__file__).with_name("requirements.txt")
    if not requirements_path.exists():
        return []
    lines = requirements_path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


setup(
    name="f1_prediction_system",
    version="0.1.0",
    description="F1 2026 Prediction System",
    packages=find_packages(exclude=("tests",)),
    install_requires=load_requirements(),
)
