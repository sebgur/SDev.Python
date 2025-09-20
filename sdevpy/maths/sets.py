


if __name__ == "__main__":
    data = ['a', 'c', 'b', 'c']
    print(data)

    # Create a set out of the data, which makes the data unique but unordered.
    # So a set cannot be accessed with indexation, and effectively its order
    # is random.
    set_data = set(data) # Make them unique, but unordered (order is random)
    print(set_data)

    # Order the set
    sorted_data = sorted(set_data)
    print(sorted_data)
