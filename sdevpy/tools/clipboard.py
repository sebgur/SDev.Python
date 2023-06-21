""" Export data to clipboard/Excel (vectors and matrices) """
import pyperclip


def concat_1dlist(data, sep='\n'):
    """ Concatenate a 1d list into a separated string """
    result = sep.join([str(d) for d in data])
    return result

def export1d(data):
    """ Export 1d list to clipboard """
    formatted = concat_1dlist(data)
    pyperclip.copy(formatted)

def concat_2dlist(data, sep='\t'):
    """ Concatenate 2d list into separated string """
    rows = []
    for d in data:
        row = concat_1dlist(d, sep=sep)
        rows.append(row)

    return concat_1dlist(rows)

def export2d(data, sep='\t'):
    """ Export 2d list to clipboard """
    formatted = concat_2dlist(data, sep=sep)
    pyperclip.copy(formatted)

if __name__ == "__main__":
    DATA = ['aer', 'b', 'c']
    # DATA = np.ones((2, 3))
    print(DATA)
    export1d(DATA)
    # data_type = type(DATA)
    # print(type(DATA))
    # if data_type is list:
    #     print("this is a list")
    # else:
    #     print("not a list")
