
repo_root="../src"
lpair=$1
cuda_device=$2
scores_file=$3

curr_score=""

score_deen () {
    echo 'Decoding Tests'
    res_dir="${1}/decoded"
    mkdir $res_dir
    suites=(4 8 9)
    scores=""
    score=""
    for n in ${suites[@]}
    do
        inp_file="../data/de-en/tests/newstest201${n}_deen.deu.tok"
        out_file="${res_dir}/newstest201${n}.out.tsv"
        ref_file="../data/de-en/tests/newstest201${n}_deen.eng"
        detok_file="${res_dir}/newstest201${n}.detok"

        CUDA_VISIBLE_DEVICES=$2, nohup rtg-decode $1 -sc -msl 512 -b 16000 -if $inp_file -of $out_file
        cat $out_file | cut -f1 | sed 's/<unk>//g' | sacremoses -l en detokenize > $detok_file
        score=$(python -m get_score -d $detok_file -r $ref_file)
        scores="${scores}\n${1}\n${score}\n\n"
    done
    curr_score=$scores
}

score_hien () {
    echo 'Decoding Tests'
    res_dir="${1}/decoded"
    mkdir $res_dir
    suites=("dev" "test")
    scores=""
    score=""
    for n in ${suites[@]}
    do
        inp_file="../data/hi-en/tests/IITB-hien_${n}-3.hin.tok"
        out_file=".${res_dir}/${n}.out.tsv"
        ref_file="../data/hi-en/tests/IITB-hien_${n}-3.eng"
        detok_file=".${res_dir}/${n}.detok"

        CUDA_VISIBLE_DEVICES=$2, nohup rtg-decode $1 -sc -msl 512 -b 16000 -if $inp_file -of $out_file
        cat $out_file | cut -f1 | sed 's/<unk>//g' | sacremoses -l en detokenize > $detok_file
        score=$(python -m get_score -d $detok_file -r $ref_file)
        scores="${scores}\n${1}\n${score}\n\n"
    done
    curr_score=$scores
}

cd $repo_root;
all_scores=""
for wdir in "${@:4}"
do
    echo $wdir
    if [ $lpair == "hi-en" ]; then
        score_hien $wdir
    else
        score_deen $wdir
    fi
    all_scores="${all_scores}/n${curr_score}"
done

echo $all_scores > $scores_file
