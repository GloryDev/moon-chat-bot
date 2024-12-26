from setuptools import setup, find_packages

setup(
    name="moon-chat-bot",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'transformers',
        'torch',
        'numpy',
    ],
    python_requires='>=3.8',
    author="Gabrial",
    author_email="GabrialEmerging@gmail.com",
    description="A chatbot using a synchronisation engine to communicate with Universal Intelligence and the User",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown"
) 