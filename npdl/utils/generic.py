# -*- coding: utf-8 -*-


def get_from_module(identifier, module_params, module_name, ):
    """Retrieves a class or function member of a modules.

    First checks `_GLOBAL_CUSTOM_OBJECTS` for `module_name`, then checks `module_params`.

    # Arguments
        identifier: the object to retrieve. It could be specified
            by name (as a string), or by dict. In any other case,
            `identifier` itself will be returned without any changes.
        module_params: the members of a modules
            (e.g. the output of `globals()`).
        module_name: string; the name of the target modules. Only used
            to format error messages.
        instantiate: whether to instantiate the returned object
            (if it's a class).
        kwargs: a dictionary of keyword arguments to pass to the
            class constructor if `instantiate` is `True`.

    # Returns
        The target object.

    # Raises
        ValueError: if the identifier cannot be found.
    """
    res = module_params[identifier]
    print(res)
    if res is None:
        raise ValueError('Invalid ' + str(module_name) + ': ' + str(identifier))
    else:
        return res

