"""
Just a mini testing file for me to try out python logic for debugging.
"""
nested_dict = {
    'key1': {'subkey1': 10, 'subkey2': 20},
    'key2': {'subkey1': 30, 'subkey2': 40},
    'key3': {'subkey1': 50, 'subkey2': 60}
}

# Use a list comprehension to extract the values of the nested dictionary
values = [value for subdict in nested_dict.values() for value in subdict.values()]