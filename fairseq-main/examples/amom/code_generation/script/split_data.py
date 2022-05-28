from tokenize import *
import random


def write_data(src,tgt,src_len,tgt_len,line,r):
    # src: where you save the downloaded data
    # tgt: where you save the splited data
    # src_len: how many lines as input
    # tgt_len: how many lines as output
    # lines, r: where you split the data
    with open(src,'a',encoding='UTF8') as f_src:
        with open(tgt,'a',encoding='UTF8') as f_tgt:
            for i in range(r+1-tgt_len,r+1):
                f_tgt.write(line[i])
                f_tgt.write(' ')
            f_tgt.write('\n')
            for i in range(r+1-tgt_len-src_len,r+1-tgt_len):
                f_src.write(line[i])
                f_src.write(' ')
            f_src.write('\n')

# generate train,test,valid subset
def getdata(path): 
    line = [] 
    with open(path, 'rb') as f:
        data = list(tokenize(f.readline))
        print(data)
        cur_string = ''
        word = []
        for item in data:
            if item.type == 62: continue 
            if item.type == 4 : 
                if cur_string != '' and cur_string.find('import') == -1 and len(word) <= 100 and len(cur_string)<=1500:
                    line.append(cur_string)
                word = []
                cur_string = ''
            else:
                if item.type == 60: continue 
                if cur_string != '' : cur_string += ' '
                s = item.string
                if item.type == 3 : s = '<str>' 
                word.append(s.strip())
                cur_string += s.strip()
        for r in range(13, len(line)):
            n = random.randint(1,10)
            if n <= 7:
                write_data('train.src', 'train.tgt', 10, 4, line, r)
            if n == 8:
                write_data('valid.src', 'valid.tgt', 10, 4, line, r)
            if n > 8:
                write_data('test.src', 'test.tgt', 10, 4, line, r)

