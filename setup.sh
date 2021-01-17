#!/bin/bash
echo 'Setting the environment for data preparation and rtg experiments'
echo 'Checking Python version : '
py_version=$(python --version)
echo $py_version
ZERO=0
if [[ $(expr "$py_version" : 'Python 3\.[7-9]') -eq $ZERO ]]
then
    echo 'Python version must be 3.7 or above'
    exit
fi

# Installing required libraries
pip3 install -r requirements.txt