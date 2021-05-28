from pathlib import Path
from setuptools import setup, find_packages

with open('README.md') as f:
    long_description = f.read()
with open('requirements.txt') as f:
    install_requires = f.read()
scripts = [str(script) for script in list(Path('scripts').iterdir())]

setup(
    name='covid19-detection',
    version='0.2.0',
    description='Detection of COVID-19 from Chest X-Ray Images',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Franco Ruggeri, Fredrik Danielsson, Muhammad Tousif Zaman, Milan Jolić',
    author_email='fruggeri@kth.se, fdaniels@kth.se, mtzaman@kth.se, jolic@kth.se',
    license='GPL',
    license_file='LICENSE',
    packages=find_packages(include=['covid19', 'covid19.*']),
    include_package_data=False,
    scripts=scripts,
    url='https://github.com/franco-ruggeri/dd2424-covid19-detection',
    install_requires=install_requires
)
