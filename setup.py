from setuptools import find_packages, setup

setup(
    name="jtk",
    packages=find_packages(include=['jtk']),
    version='0.1.0',
    description='tokenizer for nlp-waseda/roberta-base-japanese and nlp-waseda/roberta-learge-japanese',
    author='schnell',
    license='MIT',
    install_requires=['transformers', 'sentencepiece', 'pyknp'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    test_suite='tests',
)