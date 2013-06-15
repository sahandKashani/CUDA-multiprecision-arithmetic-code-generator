#!/bin/sh

generate () {
  local architecture="$1"
  local working_directory="${architecture}_results"

  rm -rf "$working_directory"
  mkdir "$working_directory"

  cd "$working_directory"
  for bits in 109 131 163 191 239; do
    mkdir "$bits"-bit
    cd "$bits"-bit
    extract 'Regs*' "$bits"
    extract 'Duration' "$bits"
    cd ..
  done
  cd ..
}

extract () {
  local field="$1"
  local bits="$2"

  rm -rf "$field"
  mkdir "$field"

  cd "$field"
  ./../../../extract.pl "$field" ../../../"$architecture"/"$bits"-bit/*.txt
  cd ..
}

generate 'kepler'
generate 'fermi'
