#!/bin/sh

generate () {
  local architecture="$1"
  local working_directory="${architecture}_results"

  rm -rf "$working_directory"
  mkdir "$working_directory"

  cd "$working_directory"
  extract 'Regs*'
  extract 'Duration'
  cd ..
}

extract () {
  local field="$1";

  rm -rf "$field"
  mkdir "$field"

  cd "$field"
  ./../../extract.pl "$field" ../../"$architecture"/*.txt
  cd ..
}

generate 'kepler'
generate 'fermi'
