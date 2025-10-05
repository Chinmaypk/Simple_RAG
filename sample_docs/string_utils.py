def reverse_string(s):
    """
    Returns the reversed version of the input string.
    """
    return s[::-1]


def count_vowels(s):
    """
    Counts the number of vowels in the input string.
    """
    vowels = "aeiouAEIOU"
    return sum(1 for ch in s if ch in vowels)


def is_palindrome(s):
    """
    Checks if a given string is a palindrome.
    """
    cleaned = "".join(ch.lower() for ch in s if ch.isalnum())
    return cleaned == cleaned[::-1]
