
# Range of variables for experiments
vocab_sizes=(1000 2000 4000 8000 16000 32000 48000)

base_exp_dir=""
base_conf_file=""
base_prep_file=""

train_src=""
train_tgt=""
valid_src=""
valid_tgt=""

make_dir () {
    echo 'Making Dir'
    if [[ ! -d $1 ]]
        then
            echo 'Creating'
            mkdir $1
    fi
}

# Change directory to bigram-bpe repo root

for sz in ${vocab_sizes[@]}
do
    exp_dir = "${base_exp_dir}/${sz}"
    make_dir ${exp_dir}
    # For baselines only conf.yml is needed.
    python -m make_conf -n conf.yml -w $exp_dir -c $base_conf_file --kwargs max_src_types=$sz max_tgt_types=$sz pieces=
    # For preparing the experiment data before hand
    python -m make_conf -n prep.yml -w $exp_dir -c $base_prep_file --kwargs max_src_types=$sz max_tgt_types=$sz
done