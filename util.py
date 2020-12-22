def label(number):
    """
    Converts an int to the correct label.
    """
    return number if number < 10 else chr(number)


def printPrediction(correct, prediction):
    """
    Prints the correct label and the predicted label.
    """
    print(f'Label: {label(correct)}\tPredicted: {label(prediction)}')
