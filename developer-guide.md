## How to setup mwe_tokenization experimentation environment ??

### Requirements :
There are 2 requirements file :
- requirements.txt : This is the current packages we are working with. It has some extra packages. Also nlcodec is default.
- requirements-new.txt : This is a trimmed list. nlcodec is packed with mwe_schemes included. Picked from (https://github.com/pegasus-lynx/nlcodec/tree/mwe_schemes).

### NOTES :
> setuptools==58.5.3 higher than this version creates issues while installing nlcodec. So, if setuptools > 58.5.3, run the following command :
```
    pip install setuptools==58.5.3
```