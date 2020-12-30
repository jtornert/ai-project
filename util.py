def label(number):
    """
    Converts an int to the correct label.
    """
    if number < 10:
        return number
    elif number < 36:
        return chr(number + 55)

    return chr(number + 61)


def invlabel(char):
    """
    Converts a character to a number that is output by the neural network.
    """
    number = ord(char)
    if number <= ord('9'):
        number = number - ord('0')
    elif number <= ord('Z'):
        number = number - ord('A') + 10
    else:
        number = number - ord('a') + 10

    return number


def printPrediction(correct, prediction):
    """
    Prints the correct label and the predicted label.
    """
    print(f'Label: {label(correct)}\tPredicted: {label(prediction)}')
