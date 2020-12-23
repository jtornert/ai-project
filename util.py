def label(number):
    """
    Converts an int to the correct label.
    """
    if number < 10:
        return number

    return chr(number + 87)


def printPrediction(correct, prediction):
    """
    Prints the correct label and the predicted label.
    """
    print(f'Label: {label(correct)}\tPredicted: {label(prediction)}')
