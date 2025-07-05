from setuptools import setup, find_packages

setup(
    name="open_iris_tracker",
    version="0.1",
    description="Sight offset detection Tracker",
    author="xin.cheng",
    packages=find_packages(),
    install_requires=[
        "opencv-python",
        "mediapipe",
        "numpy",
    ],
    entry_points={
        'console_scripts': [
            'open-iris-tracker=main:main'
        ]
    },
    python_requires=">=3.11"
)
