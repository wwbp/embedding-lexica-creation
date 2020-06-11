global {
  ducttape_output="out"
  ducttape_experimental_packages=true
  ducttape_experimental_submitters="true"
  ducttape_experimental_imports="true"
  ducttape_experimental_multiproc=true
}

import "tapes/versioners.tape"
import "tapes/submitters.tape"

import "tapes/env-clsp.tape"

import "tapes/params.tape"
import "tapes/train.tape"