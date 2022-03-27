## How to work around this repo ??

#### General concepts

- Working Directory ( work_dir / working_dir ) 
    In this repo, this term comes up often ( in various scripts that you will be using ). Working Directory here is in correspondence to that mentioned in rtg.
    In basic terms, working directory is the location, where the outputs of the script execution are stored.

    Each working directory corresponds to a different experiment / run. 

    ```
    work_dir 
        |-- _PREPARED
        |-- conf.yml                        // RTG conf file
        |-- prep.yml                        // Custom conf file to work with make_single script                      
        |---- data 
                |-- train.db                // Training File
                |-- valid.tsv.gz            // Validation File
                |-- nlcodec.src.model       // Src Vocab File
                |-- nlcodec.tgt.model       // Tgt Vocab File   
                |-- [nlcodec.shared.model]  // Shared Vocab File ( Generated when shared_vocab is true in conf)
    ```

    The tree above shows, what a working directory should contain after the data has been prepared.

#### Requirements Files:
There are 2 requirements file :
- requirements.txt : This is the current packages we are working with. It has some extra packages. Also nlcodec is default.
- requirements-new.txt : This is a trimmed list. nlcodec is packed with mwe_schemes included. Picked from (https://github.com/pegasus-lynx/nlcodec/tree/mwe_schemes).

### NOTES :
> setuptools==58.5.3 higher than this version creates issues while installing nlcodec. So, if setuptools > 58.5.3, run the following command :
```
    pip install setuptools==58.5.3
```

### Steps for setting up new environment : 

1. Create a virtualenv of your choice ( conda / virtualenv )
2. Execute `pip install setuptools==58.5.3`
3. Execute `pip install -r requirements-new.txt`

### General Steps for creating an experiment :

TBD
