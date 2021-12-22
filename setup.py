from setuptools import setup, find_packages
setup(
    name = 'MyCreditData',
    packages = find_packages(include=['Cassandra_Python_Connectivity', 'Cassandra_Python_Connectivity.*']) + find_packages(include=['common', 'common.*']) + find_packages(include=['src', 'src.*'])
    
)