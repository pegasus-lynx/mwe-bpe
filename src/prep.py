import argparse
import os
from ruamel.yaml import YAML
from pathlib import Path
from shutil import copyfile

from lib.dataset import read_parallel, Dataset
from lib.misc import FileWriter
from scripts.make_vocab import make_vocabs
from scripts.full_token import bpe_words as match_words
from scripts.prep_data import pre_process as prep_data
from scripts.prep_ngram import get_bigrams as prep_bigrams
from scripts.prep_ngram import add_bigrams as extend_vocab


def prepare_vocabs(corpus, vocab_dir, max_types, shared=False):
    bpe_vocab = { key:None for key in ['src', 'tgt', 'shared'] }
    word_vocab = { key:None for key in ['src', 'tgt', 'shared'] }
    corps = dict()
    corps['shared'] = corpus.values()
    corps['src'] = [corpus['train_src'], corpus['valid_src']]
    corps['tgt'] = [corpus['train_tgt'], corpus['valid_tgt']]
    
    if shared:
        vcb_keys = ['shared']
    else:
        vcb_keys = ['src', 'tgt']

    for key in vcb_keys:
        bpe_vocab[key] = vocab_dir / Path(f'bpe.{key}.{max_types[key]//1000}.model')
        word_vocab[key] = vocab_dir / Path(f'word.{key}.model')
        make_vocabs(corps[key], bpe_vocab[key], max_types[key], level='bpe')
        make_vocabs(corps[key], word_vocab[key], 1000000, level='word')
    return bpe_vocab, word_vocab

def prep(conf=None):
    assert conf is not None
    log('Validating configs', 1)
    conf_validation(conf)

    vdir = conf.get('vocab_dir')
    ddir = conf.get('data_dir')
    shared = conf.get('shared_vocab')
    vfilter = {
        'src': not shared,
        'tgt': not shared,
        'shared': shared
    }
    corpus = {
        'train_src': Path(conf.get('train_src')), 'train_tgt': Path(conf.get('train_tgt')),
        'valid_src': Path(conf.get('valid_src')), 'valid_tgt': Path(conf.get('valid_tgt'))
    }
    max_types = { 
        'src': conf.get('max_src_types', 10000), 
        'tgt': conf.get('max_tgt_types', 10000),
        'shared': conf.get('max_shared_types', 20000)
    }
    max_bigram_types = { 
        'src': conf.get('max_src_bigram_types', 4000), 
        'tgt': conf.get('max_tgt_bigram_types', 4000),
        'shared': conf.get('max_shared_bigram_types', 8000)
    }

    # 1. Prepare the bpe and word vocabs
    log('Preparing vocabs : word, bpe', 1)
    bpe_vocab, word_vocab = prepare_vocabs(corpus, vdir, max_types, shared=shared)

    # 2. Get the list of full tokens
    log('Preparing match word files ...', 1)
    match_word_files = dict()
    for key in vfilter.keys():
        match_word_files[key] = None
        if vfilter[key]:
            log(f'Matching words for {key}', 2)
            match_word_files[key] = vdir / Path(f'match.{key}.word.model')
            match_words(bpe_vocab[key], word_vocab[key], match_word_files[key])

    # 3. Prepare the bpe dataset
    log('Preparing base datasets ...', 1)
    log('Train Dataset', 2)
    prep_data(corpus['train_src'], corpus['train_tgt'], bpe_vocab, conf.get('truncate'), 
                conf.get('src_len'), conf.get('tgt_len'), ddir / Path('train.base.db') )
    log('Validation Dataset', 2)
    prep_data(corpus['valid_src'], corpus['valid_tgt'], bpe_vocab, conf.get('truncate'), 
                conf.get('src_len'), conf.get('tgt_len'), ddir / Path('valid.base.tsv') )

    # 4. Find and add the bigrams to the vocab files
    log('Reading parallel datafile', 1)
    parallel_dataset = Dataset(['src', 'tgt'])
    parallel_dataset.add(read_parallel(ddir / Path('train.base.tsv')))
    parallel_dataset.add(read_parallel(ddir / Path('valid.base.tsv')))

    log('Preparing extended vocabs ...', 1)
    mod_vocab = dict()
    for key in vfilter.keys():
        mod_vocab[key] = None
        if vfilter[key]:
            log(f'Preparing bigram for {key}', 2)
            bigrams = prep_bigrams(match_word_files[key], [parallel_dataset.lists.values() if key == 'shared' \
                                    else parallel_dataset.lists[key]], min_freq=conf.get('min_freq', 50))
            extend_vocab(bpe_vocab[key], bigrams, vdir, max_add=max_bigram_types[key])
            mod_vocab[key] = vdir / Path(str(bpe_vocab[key].name).replace('.model', '.mod.model'))

    # 5. Prepare the final dataset
    log('Preparing final datasets ...', 1)
    log('Train Dataset', 2)
    prep_data(corpus['train_src'], corpus['train_tgt'], mod_vocab, conf.get('truncate'), 
                conf.get('src_len'), conf.get('tgt_len'), ddir / Path('train.db') )
    log('Valid Dataset', 2)
    prep_data(corpus['valid_src'], corpus['valid_tgt'], mod_vocab, conf.get('truncate'), 
                conf.get('src_len'), conf.get('tgt_len'), ddir / Path('valid.tsv') )

def prep_working_dir(work_dir, conf_file):
    if not work_dir.exists():
        log('Making work dir', 1)
        work_dir.mkdir()
    if conf_file.parent != work_dir:
        new_conf_file = work_dir / Path('prep.yml')
        log(f'Copying conf file : {conf_file}', 1)
        copyfile(conf_file, new_conf_file)
        conf_file = new_conf_file
    return work_dir, conf_file

def prep_subdirs(work_dir):
    vocab_dir = work_dir / Path('_vocabs/')
    if not vocab_dir.exists():
        log('Making vocab directory', 1)
        vocab_dir.mkdir()
    data_dir = work_dir / Path('data/')
    if not data_dir.exists():
        log('Making data directory', 1)
        data_dir.mkdir()
    return vocab_dir, data_dir

def args_validation(args):
    assert args.work_dir is not None
    assert args.work_dir.is_dir()
    if args.config_file is not None:
        assert args.config_file.exists()
    else:
        assert args.work_dir.exists()
        assert Path(args.work_dir / 'prep.yml').exists()

def read_conf(conf_file):
    yaml = YAML(typ='safe')
    if type(conf_file) == str:
        conf_file = Path(conf_file)
    return yaml.load(conf_file)

def save_meta(conf, work_dir):
    # print(conf)
    fw = FileWriter(work_dir / Path('meta.txt'))
    fw.heading(f'DATA PREP RUN : {work_dir.name}')
    fw.section('Directories :', [f'{key} : {value}' for key,value in conf.items() if 'dir' in key])
    fw.section('Vocab Parameters : ', [f'{key} : {value}' for key, value in conf.items() if 
                                            'dir' not in key and 'train' not in key and 'valid' not in key])
    fw.section('Corpus : ', [f'{key} : {value}' for key,value in conf.items() if 'train' in key or 'valid' in key])
    fw.close()

def conf_validation(conf):
    for x in ['shared_vocab', 'train_src', 'train_tgt', 'valid_src', 'valid_tgt']:
        assert x in conf.keys()
    
    for x in ['src_len', 'tgt_len']:
        if not x in conf.keys():
            conf[x] = 512

    if not 'truncate' in conf.keys():
        conf['truncate'] = True
    
    if not 'min_bigram_freq' in conf.keys():
        conf['min_bigram_freq'] = 10

    for key in ['src', 'tgt', 'shared']:
        if not f'max_{key}_types' in conf.keys():
            conf[f'max_{key}_types'] = 20000 if key == 'shared' else 10000
        if not f'max_{key}_bigram_types' in conf.keys():
            conf[f'max_{key}_bigram_types'] = 8000 if key == 'shared' else 4000

def log(text, ntabs=0):
    print(f"{'    '*ntabs} > {text}")

def parse_args():
    parser = argparse.ArgumentParser('src.prep', description='Script to preapre the baseline and bi-gram data.')
    parser.add_argument('-w', '--work_dir', type=Path, help='Path of the working directory.')
    parser.add_argument('-c', '--config_file', type=Path, help='''Path to the config file. If not provided  
                                                the script searches for the conf file in work dir''')
    return parser.parse_args()

def main():
    log('Starting Script')
    args = parse_args()
    log('Args loaded')
    args_validation(args)
    log('Args validated')
    
    conf_file = args.work_dir / Path('prep.yml')
    if args.config_file is not None:
        conf_file = args.config_file
    work_dir, conf_file = prep_working_dir(args.work_dir, conf_file)
    log(f'Preparing work_dir : {work_dir} ...')
    
    log(f'Preparing sub dirs : _vocabs, data ...')
    vocab_dir, data_dir = prep_subdirs(work_dir)
    
    log(f'Setting conf file : {conf_file}')
    conf = read_conf(conf_file)
    conf['vocab_dir'] = vocab_dir
    conf['data_dir'] = data_dir
    
    log('Starting data preparation ...')
    prep(conf=conf)

    os.system(f'touch {work_dir}/_PREPARED')

    log('Saving the meta file for the prep')
    save_meta(conf, work_dir)


if __name__ == '__main__':
    main()