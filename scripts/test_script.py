def get_inputs(text, expect = None, default=None):
    ret = None
    while ret is None or ret == "":
        if expect is None:
            ret = input(text + " : ")
        else:
            while ret not in expect:
                ret = input(text + " " + str(expect) + " : ")

        if default is not None and ret is None or ret == "":
            ret = default
            break

    if ret is None:
        return default
    return ret


def folderChecker():
    model_code = get_inputs("Input a number")
    if model_code in ["1","2","3","4"]:
        response = get_inputs("model code already present in the API. Would you like to update, replace existing folder or neither?", ['update','replace', 'neither'])
        if response == 'neither':
            model_code, response = folderChecker()
        return model_code, response

    else:
        response='all good'
        return model_code, response

m, r = folderChecker()
print(m)
print(r)