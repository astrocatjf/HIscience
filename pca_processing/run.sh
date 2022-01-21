#!/bin/bash

for i in {1..100}
do
  # Print the sky number
  echo "$i"
  python pca_script.py 1 "$i"
done
