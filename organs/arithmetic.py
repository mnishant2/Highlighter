'''
The purpose of this module is to contain the arithmetic functions used in the entire application

'''

# This function takes in two numbers and tells whether are equivalent
def equivalent(n1, n2):
    n1abs = abs(n1);
    n2abs = abs(n2)
    if n1abs/n2abs < 1.19 and n1abs/n2abs > 0.81:
        return True
        pass
    return False
    pass


