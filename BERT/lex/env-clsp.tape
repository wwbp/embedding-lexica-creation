package rg :: .versioner=disk .path="/home/zwu49/ztwu/empathy_dictionary/BERT" .rev=NULL {}
package data :: .versioner=disk .path="/home/zwu49/ztwu/empathy_dictionary/data" .rev=NULL {}

global {
  grid="sge"
  gpuId="\$(free-gpu)"
  pyenv="conda:nlp"

  cpuActionFlags=""
#  cpuResourceFlags="-l 'mem_free=30g,ram_free=30g'"
  cpuResourceFlags="-l 'hostname=b*,mem_free=30g,ram_free=30g'"

  gpuActionFlags="-q g.q"
  gpuResourceFlags="-l 'hostname=b1[1235679]*|c*,gpu=1,mem_free=30g,ram_free=40g'"
#  gpuResourceFlags="-l 'hostname=b19,gpu=1,mem_free=20g,ram_free=20g'"
}