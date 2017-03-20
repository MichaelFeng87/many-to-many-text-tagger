


from re import sub
from nltk import word_tokenize

def preprocess_string(string_data):
    assert isinstance(string_data, (str, unicode)), 'Got %s' % type(string_data)
    
    string_data = string_data.lower().strip()
    
    #memperbaiki kegagalan decoding dash '-'
    string_data = sub(r'&#2013266070','-',string_data)
    string_data = sub(r'&#2013266069',':',string_data)
    
    #memperbaiki kegagalan decoding '\n'
    string_data = sub(r'\\n',' ',string_data)

    #memperbaiki posisi tanda kurung
    string_data = sub(r'([a-zA-Z0-9])(\()',r'\1 \2',string_data)
    string_data = sub(r'(\))([a-zA-Z0-9])',r'\1 \2',string_data)

    # mengganti - dengan <DASH>
    string_data = sub(r'\-{1}',' <DASH> ', string_data)
    string_data = sub(r'\:{1}',' <COLON> ', string_data)
    
    #memperbaiki kemungkinan kuhp dan kuhap yang berbeda format
    string_data = sub(r'(k\.u\.h\.p)','kuhp',string_data)
    string_data = sub(r'(k\.u\.h\.a\.p)','kuhap',string_data)
    string_data = sub(r'(kuhpidana)','kuhp',string_data)
    string_data = sub(r'(kuh pidana)','kuhp',string_data)
    string_data = sub(r'(kuh\.pidana)','kuhp',string_data)
    string_data = sub(r'([0-9\(])(kuhp)', r'\1 \2',string_data)
    string_data = sub(r'(kuhp)([0-9\)])', r'\1 \2',string_data)
    string_data = sub(r'([0-9\(])(kuhap)', r'\1 \2',string_data)
    string_data = sub(r'(kuhap)([0-9\)])', r'\1 \2',string_data)
    
    #memperbaiki kemungkinan format uu
    string_data = sub(r'(u\.u\.)','uu',string_data)

    #memperbaiki format tulisan    
    string_data = sub(r'((\s)(yat)(\s))',' ayat ',string_data)
    string_data = sub(r'((\s)(cara)(\s))',' acara ',string_data)
    
    # mengganti . dengan <DOT>
    string_data = sub(r'\.{1}', ' <DOT> ', string_data)

    # menganti format pasal yang berdempetan dengan huruf dengan spasi, pasal362 -> pasal 362 dan 
    # menggunakan token number
    string_data_tokenized = sub(r'(\d+)',r' <NUMBER> ', string_data)
    string_data_real = sub(r'(\d+)',r' \1 ', string_data)

    #tokenize tiap kata
    string_data_tokenized = string_data_tokenized.split()
    string_data_real =string_data_real.split()
    return string_data_tokenized,string_data_real