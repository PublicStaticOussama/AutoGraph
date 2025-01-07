from setuptools import setup, find_packages

setup(
    name='AutoGraph',
    version='0.1.0',
    packages=find_packages(),  # Automatically find and include all packages in the library
    install_requires=[
        'numpy',
    ],
    include_package_data=True,  # Include non-code files specified in MANIFEST.in
    description='Automatic differentiation and DeepLearning Engine using Numpy only',
    long_description=open('README.md').read(),  # Use README file as long description
    long_description_content_type='text/markdown',
    author='Wahbi Oussama',
    author_email='oussama.wahbi420@gmail.com',
    url='https://github.com/PublicStaticOussama/AutoGraph',  # URL of the project
    license='MIT',  # License type
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.7',  # Minimum Python version required
)
