## Common Commands :

All these commands work from `$repo_root/src`

1. Making prep.yml : 

```
    python -m make_conf -n prep.yml -w <work_dir> -c <path_to_base_config> --kwargs max_src_types=<size> max_tgt_types=<size>
```

2. Making conf.yml :
```
    python -m make_conf -n conf.yml -w <work_dir> -c ../configs/base/hi-en/base.conf.yml --kwargs src_vocab=<size> tgt_vocab=<size> max_src_vocab=<size> max_tgt_vocab=<size> pieces=bpe
```

3. Running the experiment :
```
    rtg-pipe <path_to_work_dir> -G
```
As of now this will be enough to get the text decodec.
After decoding we can evaluate it separately. Working on that.