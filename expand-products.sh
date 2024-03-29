#!/bin/bash

# Safely execute this bash script
# e exit on first failure
# x all executed commands are printed to the terminal
# u unset variables are errors
# a export all variables to the environment
# E any trap on ERR is inherited by shell functions
# -o pipefail | produces a failure code if any stage fails
set -Eeuoxa pipefail

# Get the directory of this script
LOCAL_DIRECTORY="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# build
$LOCAL_DIRECTORY/scripts/build.sh

docker run -v ~/.powerml:/root/.powerml \
    -v ~/.lamini:/root/.lamini \
    -v $LOCAL_DIRECTORY/data:/app/shopper/data \
    -e LAMINI_API_KEY=$LAMINI_API_KEY \
    -it --rm --entrypoint /app/shopper/scripts/start-expand-products.sh shopper:latest "$@"

