import numpy as np

class NotFittedError(ValueError, AttributeError):
    """

    Helper class for not fitted model

    """
    pass

def check_fit(obj):
    """
    check if 'fit' method has been called for this particular linear regression
    instance before attempting to do prediction.

    :param obj: a linear regression instance
    :return: raise a NotFittedError error message
    """
    if not hasattr(obj, 'tetha'):
        msg = ("This linear regression instance has not been fitted yet. "
               "Please call the 'fit' attribute to proceed with the regression.")
        raise NotFittedError(msg )

def check_is_array(*arrs):
    """
    check if the inputs is either numpy arrays or list objects. If input is list,
    cast it onto a numpy array data.

    :param arrs: argument list of objects to be checked
    :return: tuple of the input objects with type numpy array. Raise a type error
             if the inputs are neither of list or array type.
    """
    arrays = []
    for arr in arrs:
        if isinstance(arr, list) or isinstance(arr, np.ndarray):
            arr = np.array(arr)
            arrays.append(arr)
        else:
            raise TypeError("Expected a list or numpy array object, received {0} type data".format(type(arr).__name__))

    return tuple(arrays)
