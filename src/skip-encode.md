## Skip Encoding

Okay, so to discuss the current approach:

#### Representing the skipgrams

The skipgrams are represneted in the following form : 
$$
    a_ <skip_char> ... b_ 
$$

#### Making the vocab trie

The vocab trie is made the similar way as in the case of BPE Scheme

#### Process 

The encoding starts normally. However, when we find a skip_char somewhere in the trie then we call the get_skips functions

The get_skips function will return the list of tokens that can be replaced for the skip chars.
And before returning the things to the main encode function we check that if the current token is skippd then id the skippability a valid case and if not then the candidate skip tokens are not returned.

This is the complete process that we follow. 

Now what are the changes that we can mkae to make the code more efficient. First the adding of the endswith does make it more efficient but what it does is that, it limits the use case for skipgrams that end with a space char. Earlier, although not that efficient it used to check for every available token and then we could go forward with any kind of skipgrams. 

So how to go about writing the get_skips function ??