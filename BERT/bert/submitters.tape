# COMMANDS: the bash commands from some task
# TASK, REALIZATION, CONFIGURATION: variables passed by ducttape
submitter shell :: COMMANDS TASK_VARIABLES {
  action run {
    set +u  # needed to fix a virtualenv bug
    if [[ ! -z ${pyenv:-} ]]; then
      virtualenv=$pyenv

      # Load the environment
      if [[ $virtualenv == conda:* ]]; then
#        . /etc/profile.d/conda.sh
        . ~/.bashrc
        target=$(echo $virtualenv | cut -d: -f2-)
        source deactivate
        conda activate $target
      else
        source $virtualenv
      fi
    fi
    set -u

    STARTED=$(date +%s)
    time eval "$COMMANDS"
    STOPPED=$(date +%s)
    TIME=$(($STOPPED - $STARTED))
    echo $TIME > ducttape_time.txt
    set -u
  }
}

# COMMANDS: the bash commands from some task
# TASK, REALIZATION, CONFIGURATION: variables passed by ducttape
submitter sge :: action_flags
              :: COMMANDS
              :: TASK REALIZATION TASK_VARIABLES CONFIGURATION {
  action run {
    wrapper="ducttape_sge_job.sh"
    echo "#!/usr/bin/env bash" >> $wrapper
    echo "" >> $wrapper
    echo "#$ $resource_flags" >> $wrapper
    echo "#$ $action_flags" >> $wrapper
    echo "#$ -j y" >> $wrapper
    echo "#$ -o localhost:$PWD/job.out" >> $wrapper
    echo "#$ -N $TASK-$REALIZATION" >> $wrapper
    echo "" >> $wrapper

    # Bash flags aren't necessarily passed into the scheduler
    # so we must re-initialize them

    echo "set -euo pipefail" >> $wrapper
    echo "" >> $wrapper
    echo "$TASK_VARIABLES" | perl -pe 's/=/="/; s/$/"/' >> $wrapper

    # Setup the virtual environment
    cat >> $wrapper <<EOF

set +u  # needed to fix a virtualenv bug
if [[ ! -z \${pyenv:-} ]]; then
  virtualenv=\$pyenv

  # Load the environment
  if [[ \$virtualenv == conda:* ]]; then
    . ~/.bashrc
    target=\$(echo \$virtualenv | cut -d: -f2-)
    source deactivate
    conda activate \$target
  elif [[ \$virtualenv == vir:* ]]; then
    target=\$(echo \$virtualenv | cut -d: -f2-)
    source ~/\$target/bin/activate
  else
    source \$virtualenv
  fi
fi
set -u

EOF

    # The current working directory will also be changed by most schedulers
    echo "cd $PWD" >> $wrapper

    echo >> $wrapper
    echo "echo \"HOSTNAME: \$(hostname)\"" >> $wrapper
    echo "echo" >> $wrapper
    echo "echo CUDA in ENV:" >> $wrapper
#    echo "env | grep CUDA" >> $wrapper
    echo "env | grep SGE" >> $wrapper
    echo >> $wrapper

    echo "$COMMANDS" >> $wrapper
    echo "echo \$? > $PWD/exitcode" >> $wrapper  # saves the exit code of the inner process

    # Use SGE's -sync option to prevent qsub from immediately returning
    sleep $(($RANDOM % 300))
    qsub -V -S /bin/bash $wrapper | grep -Eo "Your job [0-9]+" | grep -Eo "[0-9]+" > $PWD/job_id
    job_id=`cat $PWD/job_id`

    # async job killer
    exitfn () {
      trap SIGINT
      echo "wait until I kill the job $job_id"
      qdel $job_id
      exit
    }

    trap "exitfn" INT

    # don't use -sync y, instead, wait on exitcode
    while [ ! -z "`qstat -u $USER | grep $job_id`" ]
    do
      sleep 15
    done

    trap SIGINT

    # restore the exit code saved from the inner process
    EXITCODE=$(cat $PWD/exitcode)
    [ $EXITCODE = "0" ]
  }
}
