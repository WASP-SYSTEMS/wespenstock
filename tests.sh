#!/bin/bash

# delay to let Dind start up...
sleep 30

# check that Dind has access to the /workdir
docker run --rm -v $PWD:/workdir --workdir /workdir debian bash -c 'ls -al'

pip install -e '.[dev]'
python3 -m pytest --tb=native --cov=crs --cov=crscommon
