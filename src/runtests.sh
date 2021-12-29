
# Range of variables for experiments
vocab_sizes=(2000 4000 8000 16000)

repo_root=""
base_exp_dir="../../exps/hi-en/baseline"
base_conf_file="../configs/base/base.conf.yml"
base_prep_file=""

train_src=""
train_tgt=""
valid_src=""
valid_tgt=""

pieces="unigram"
codec_lib="sentpiece"

cuda_device="2"

make_dir () {
    echo 'Making Dir'
    if [[ ! -d $1 ]]
        then
            echo 'Creating'
            mkdir $1 -p
    fi
}

# Change directory to bigram-bpe repo root. This will allow to run the script from outside the repo.
cd $repo_root

for sz in ${vocab_sizes[@]}
do
    exp_dir="${base_exp_dir}/${sz}"
    mkdir ${exp_dir} -p
    # For baselines only conf.yml is needed.
    python -m make_conf -n conf.yml -w $exp_dir -c $base_conf_file --kwargs src_vocab=$sz tgt_vocab=$sz max_src_types=$sz max_tgt_types=$sz pieces=$pieces codec_lib=$codec_lib
    # For preparing the experiment data before hand
    #python -m make_conf -n prep.yml -w $exp_dir -c $base_prep_file --kwargs max_src_types=$sz max_tgt_types=$sz
    CUDA_VISIBLE_DEVICES=$cuda_device, rtg-pipe $exp_dir -G
    #python -m get_score -d ../../exps/test_step199000_beam4_ens10_lp0.6/dev-IITB.out.mosesdetok -r ../data/hi-en/tests/IITB-hien_dev-3.eng
done
