global {
  ducttape_output="roberta"
  ducttape_experimental_packages=true
  ducttape_experimental_submitters="true"
  ducttape_experimental_imports="true"
  ducttape_experimental_multiproc=true
}

import "bert/versioners.tape"
import "bert/submitters.tape"

import "bert/env-clsp.tape"

import "bert/params.tape"
import "bert/train.tape"