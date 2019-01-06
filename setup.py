import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
        name='ml_reusable',
        version='0.1',
        author="Erik Ekstedt",
        author_email="eeckee@gmail.com",
        description="Stuff I tend to reinvent for myself",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/ErikEkstedt/ml_reusable",
        packages=setuptools.find_packages(),
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: Linux",
            ],
)
