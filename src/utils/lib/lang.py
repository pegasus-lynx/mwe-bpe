class CharTokens(object):
    symbols = [
        '.', ',', '(', ')', '[', ']', '{', '}', '!',
        ':', '-', '"', "'", ';', '<', '>', '?', '&',
        '–', '@', ' ', '\t', '\n'
    ]
    en_digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    hi_digits = ['०', '१', '२', '३', '४', '५', '६', '७', '८', '९']
    en_chars  = [
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
        'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
        'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
    ]
    hi_chars  = [
        'ऀ', 'ँ', 'ं', 'ः', 'ऄ', 'अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ऋ', 'ऌ', 
        'ऍ', 'ऎ', 'ए', 'ऐ', 'ऑ', 'ऒ', 'ओ', 'औ', 'क', 'ख', 'ग', 'घ', 'ङ', 
        'च', 'छ', 'ज', 'झ', 'ञ', 'ट', 'ठ', 'ड', 'ढ', 'ण', 'त', 'थ', 'द', 
        'ध', 'न', 'ऩ', 'प', 'फ', 'ब', 'भ', 'म', 'य', 'र', 'ऱ', 'ल', 'ळ', 
        'ऴ', 'व', 'श', 'ष', 'स', 'ह', 'ऺ', 'ऻ', '़', 'ऽ', 'ा', 'ि', 'ी', 'ु', 
        'ू', 'ृ', 'ॄ', 'ॅ', 'ॆ', 'े', 'ै', 'ॉ', 'ॊ', 'ो', 'ौ', '्', 'ॎ', 'ॏ', 'ॐ', 
        '॑', '॒', '॓', '॔', 'ॕ', 'ॖ', 'ॗ', 'क़', 'ख़', 'ग़', 'ज़', 'ड़', 'ढ़', 'फ़', 
        'य़', 'ॠ', 'ॡ', 'ॢ', 'ॣ', '।', '॥', '॰', 'ॱ', 'ॲ', 'ॳ', 'ॴ', 'ॵ', 
        'ॶ', 'ॷ', 'ॸ', 'ॹ', 'ॺ', 'ॻ', 'ॼ', 'ॽ', 'ॾ', 'ॿ'
    ]

    @classmethod
    def eng(cls, text:str):
        for x in text:
            digit = x in cls.en_digits
            letter = x in cls.en_chars
            symbol = x in cls.symbols
            valid = (digit or letter or symbol)
            if not valid:
                return False
        return True

    @classmethod
    def hin(cls, text:str):
        for x in text:
            digit = x in cls.hi_digits
            digit = digit or (x in cls.en_digits)
            letter = x in cls.hi_chars
            symbol = x in cls.symbols
            valid = (digit or letter or symbol)
            if not valid:
                return False
        return True
