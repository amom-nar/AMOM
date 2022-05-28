def ld(str1,str2):
    m,n = len(str1)+1,len(str2)+1
    matrix = [[0]*n for i in range(m)]
    matrix[0][0] = 0
    for i in range(1,m):
        matrix[i][0] = matrix[i-1][0] + 1
    for j in range(1,n):
        matrix[0][j] = matrix[0][j-1]+1  
    for i in range(1,m):
        for j in range(1,n):
            if str1[i-1] == str2[j-1]:
                matrix[i][j] = matrix[i-1][j-1]
            else:
                matrix[i][j] = min(matrix[i-1][j-1],matrix[i-1][j],matrix[i][j-1])+1
    
    return 1-matrix[m-1][j-1]/(m+n)

def ES(input_file1,input_file2):
    with open(input_file1, 'r', encoding='utf-8') as Src:
        with open(input_file2,'r',encoding='utf-8') as Tgt:
            src = Src.readlines()
            tgt = Tgt.readlines()
            tot = 0
            for i in range(len(src)):
                src[i] = src[i].strip()
                tgt[i] = tgt[i].strip()
                s = src[i].split(' ')
                t = tgt[i].split(' ')
                tot += ld(s,t)
            print(tot/len(src))

if __name__ == '__main__':
    ES("ref","sys")