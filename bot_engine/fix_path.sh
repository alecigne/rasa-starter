#!/bin/bash

# Fixing the path in Git Bash after pipenv shell
# Usage: source fix_path.sh OR . fix_path.sh

export PATH=$(echo $PATH | sed 's_C:/Users_/c/Users_g')