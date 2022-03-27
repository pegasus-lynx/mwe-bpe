
# Range of variables for experiments
# vocab_sizes=(4000 8000)
vocab_sizes=(8800)

repo_root="../src"
base_exp_dir="../temp/trials/extmwe-mar28/hi-en/"
base_conf_file="../configs/base/hi-en/base.conf.yml"
base_prep_file="../configs/base/hi-en/base.prep.ext.yml"

shared=0
split_ratio=0.25

pieces="extmwe"
src_pieces="extmwe"
tgt_pieces="extmwe"

mwe_tokens="bi,tri,ski"

max_mwes=800

cuda_device="2"

decode_tests_deen () {
    echo 'Decoding Tests'
    res_dir="${1}/results"
    mkdir $res_dir
    suites=(4 8 9)

    for n in ${suites[@]}
    do
        inp_file="../data/de-en/tests/newstest201${n}_deen.deu"
        out_file="${res_dir}/newstest201${n}.out.tsv"
        ref_file="../data/de-en/tests/newstest201${n}_deen.eng"
        detok_file="${res_dir}/newstest201${n}.detok"

        CUDA_VISIBLE_DEVICES=$2, nohup rtg-decode $1 -sc -msl 512 -b 8000 -if $inp_file -of $out_file
        cat $out_file | cut -f1 | sed 's/<unk>//g' | sacremoses -l en detokenize > $detok_file
        python -m get_score -d $detok_file -r $ref_file
    done
}

decode_tests_hien () {
    echo 'Decoding Tests'
    res_dir="${1}/results"
    mkdir $res_dir
    suites=("dev" "test")

    for n in ${suites[@]}
    do
        inp_file="../data/hi-en/tests/IITB-hien_${n}-3.hin"
        out_file=".${res_dir}/${n}.out.tsv"
        ref_file="../data/hi-en/tests/IITB-hien_${n}-3.eng"
        detok_file=".${res_dir}/${n}.detok"

        CUDA_VISIBLE_DEVICES=$2, nohup rtg-decode $1 -sc -msl 512 -b 8000 -if $inp_file -of $out_file
        cat $out_file | cut -f1 | sed 's/<unk>//g' | sacremoses -l en detokenize > $detok_file
        python -m get_score -d $detok_file -r $ref_file
    done
}

# Change directory to bigram-bpe repo root. This will allow to run the script from outside the repo.
cd $repo_root

for sz in ${vocab_sizes[@]}
do
    exp_dir="${base_exp_dir}/${sz}"
    mkdir ${exp_dir} -p
    
    # 1. For preparing the experiment data before hand
    if [ $shared -eq 1 ]; then
        python -m make_conf -n prep.yml -w $exp_dir -c $base_prep_file -r $repo_root --kwargs pieces=$pieces max_types=$sz mwe_tokens=$mwe_tokens split_ratio=$split_ratio max_mwes=$max_mwes
    else
        python -m make_conf -n prep.yml -w $exp_dir -c $base_prep_file -r $repo_root --kwargs max_src_types=$sz max_tgt_types=$sz src_pieces=$src_pieces tgt_pieces=$tgt_pieces mwe_tokens=$mwe_tokens split_ratio=$split_ratio max_mwes=$max_mwes
    fi

    # 2. Prepare the data
    python -m make_ext -w $exp_dir -c "${exp_dir}/prep.yml"

    # 3. For baselines only conf.yml is needed.
    if [ $shared -eq 1 ]; then
        python -m make_conf -n conf.yml -w $exp_dir -c $base_conf_file -r $repo_root --kwargs src_vocab=$sz tgt_vocab=$sz max_types=$sz
    else
        python -m make_conf -n conf.yml -w $exp_dir -c $base_conf_file -r $repo_root --kwargs src_vocab=$sz tgt_vocab=$sz max_src_types=$sz max_tgt_types=$sz
    fi

    # 4. Running experiments
    # CUDA_VISIBLE_DEVICES=$cuda_device, rtg-pipe $exp_dir -G

    # 5. Decode tests

    # if [ $shared -eq 1 ]; then
    #     decode_tests_deen $exp_dir $cuda_device
    # else
    #     decode_tests_hien $exp_dir $cuda_device
    # fi
done
