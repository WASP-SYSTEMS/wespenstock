#!/bin/sh
BEAR_USE_PRELOAD=1 bear --append --output /work/compile_commands.json -- /bin/bash /usr/local/bin/compile || bear --append --cdb /work/compile_commands.json /bin/bash /usr/local/bin/compile
