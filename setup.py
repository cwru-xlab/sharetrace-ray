import setuptools

setuptools.setup(
    name="contact-tracing",
    version="0.1.0",
    packages=setuptools.find_packages(),
    install_requires=["pymetis @ git+https://git@github.com/inducer/pymetis"],
    url="https://github.com/csds438-f21-project/contact-tracing",
    license="MIT",
    author="Ryan Tatton",
    author_email="ryan.tatton@case.edu",
    description="Improved implementation of ShareTrace contact tracing",
)
