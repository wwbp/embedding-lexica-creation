global {
  ducttape_output="generate_rerun_ps"
  ducttape_experimental_packages=true
  ducttape_experimental_submitters="true"
  ducttape_experimental_imports="true"
  ducttape_experimental_multiproc=true
}

import "lex/versioners.tape"
import "lex/submitters.tape"

import "lex/env-clsp.tape"

import "lex/params.tape"
import "lex/train.tape"