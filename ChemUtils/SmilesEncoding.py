import numpy as np

char_list= ["H","C","N","O","F","P","S","Cl","Br","I",
"n","c","o","s",
"1","2","3","4","5","6","7","8","9",
"(",")","[","]",
"-","=","#","/","\\","+","@"]

CHAR_LEN = len(char_list)

char_dict={'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'P': 5, 
'S': 6, 'Cl': 7, 'Br': 8, 'I': 9, 
'n': 10, 'c': 11, 'o': 12, 's': 13, 
'1': 14, '2': 15, '3': 16, '4': 17, '5': 18, '6': 19, '7': 20, '8': 21, '9':22,
'(': 23, ')': 24, '[': 25, ']': 26, '-': 27, '=': 28, '#': 29, 
'/': 30, '\\': 31, '+': 32, '@': 33}

def smiles_to_hot(smiles):
    nxt = False
    hot_smiles = []
    for i in range(len(smiles)):
        if nxt:
            nxt = False
            continue
            
        if smiles[i:i+2] in char_list:
            nxt = True
            hot_smiles.append(char_dict[smiles[i:i+2]])
        elif smiles[i] in char_list:
            hot_smiles.append(char_dict[smiles[i]])
        else:
            return None
    
    return hot_smiles


def hot_to_smiles(hot_smiles):
    smiles = ''
    for i in hot_smiles:
        smiles += char_list[i]
    return smiles


def array_to_smiles(np_array, end_token):
    smiles = []
    for i,x in enumerate(np_array):
        end_pos = np.where(x==end_token)[0]
        if len(end_pos)==0:
            end_pos = len(x)
        else:
            end_pos = end_pos[0]
            
        smiles.append(hot_to_smiles(x[1:end_pos]-1))
        
    return smiles