from setuptools import setup, find_packages

setup(
    name='liquidstatemachines',  # Name des Pakets
    version='0.1.0',  # Versionsnummer
    description='Brian 2 implementation of the liquid state machine',
    author='Karl-Augustin Jahnel',
    author_email='dein.email@example.com',
    packages=find_packages(),  # Automatisch alle Unterpakete finden
    install_requires=[],  # Abhängigkeiten, falls nötig
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Deine Lizenz, z.B. MIT
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Mindest-Python-Version
)
