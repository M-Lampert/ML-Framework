# This is necessary to install it via pip
import setuptools

with open("README.md", "r") as fh:
    description = fh.read()

setuptools.setup(
    name="framework",
    version="1.0",
    author="Team_17",
    packages=["framework"],
    description="A neural network framework",
    long_description=description,
    long_description_content_type="text/markdown",
    url="https://gitlab2.informatik.uni-wuerzburg.de/hci/teaching/courses/machine-learning/student-submissions/ws21/Team-17/framework.git",
    license="MIT",
    python_requires=">=3.9",
    install_requires=["numpy", "tqdm", "matplotlib", "pandas"],
)
