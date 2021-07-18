from lib.schemes import load_scheme, SkipScheme

from nlcodec import Type, codec
from pathlib import Path

work_dir = Path("../data.simp/runs.single/test-sgram-freq-encoding-skip/")

vocab_files = {
    'src': work_dir / Path('data/nlcodec.src.model'),
    'tgt': work_dir / Path('data/nlcodec.tgt.model')
}

keys = ['src', 'tgt']

codecs = {}

scheme = load_scheme('skipgram')

for key in keys:
    table, _ = Type.read_vocab(vocab_files[key])
    codecs[key] = scheme(table)

texts = [
    'She ( her ) ( things here )'
]

def dash():
    print('-'*80)

def cprint(*args):
    for arg in args:
        if type(arg) == str and arg == 'dash':
            dash()
        else:
            print(arg)

for text in texts:
    encoded = codecs['tgt'].encode(text)
    parts = [codecs['tgt'].table[x].name for x in encoded]
    decoded = codecs['tgt'].decode_str(parts)
    cprint('dash', text, 'dash', encoded, parts, 'dash', decoded, 'dash')
    print()
