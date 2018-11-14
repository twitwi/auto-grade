
from threading import Timer


def debounce(wait, argument_dependent=False):
    """ Decorator that will postpone a functions
        execution until after wait seconds
        have elapsed since the last time it was invoked. """
    def decorator(fn):
        def debounced(*args, **kwargs):
            def call_it():
                fn(*args, **kwargs)
            if not hasattr(debounced, 't'):
                debounced.t = {}
            if argument_dependent:
                _hash = str(hash(str(args)))+str(hash(str(kwargs)))
            else:
                _hash = "-"
            try:
                debounced.t[_hash].cancel()
            except(KeyError):
                pass
            debounced.t[_hash] = Timer(wait, call_it)
            debounced.t[_hash].start()
        return debounced
    return decorator
