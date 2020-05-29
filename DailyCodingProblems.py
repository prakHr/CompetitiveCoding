#22
def get_sentence_split(s, words):
    if not s or not words:
        return []

    word_set = set(words)
    sentence_words = list()
    for i in range(len(s)):
        if s[0:i + 1] in word_set:
            sentence_words.append(s[0:i + 1])
            word_set.remove(s[0:i + 1])
            sentence_words += get_sentence_split(s[i + 1:], word_set)
            break

    return sentence_words


assert get_sentence_split("thequickbrownfox", ['quick', 'brown', 'the', 'fox']) == [
    'the', 'quick', 'brown', 'fox']
assert get_sentence_split("bedbathandbeyond", [
                          'bed', 'bath', 'bedbath', 'and', 'beyond']) == ['bed', 'bath', 'and', 'beyond']

#27
brace_map = {
    ")": "(",
    "}": "{",
    "]": "["
}


def is_balanced(s):
    stack = list()
    for char in s:
        if stack and char in brace_map and stack[-1] == brace_map[char]:
            stack.pop()
        else:
            stack.append(char)
    return not stack

#50
class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


def solve_graph(root):
    if root.val.isnumeric():
        return float(root.val)

    return eval("{} {} {}".format(solve_graph(root.left), root.val, solve_graph(root.right)))

#56
def can_color_graph(adjacency_matrix, k):
    max_adjacencies = 0
    for row in adjacency_matrix:
        max_adjacencies = max(max_adjacencies, sum(row))

    return k > max_adjacencies


adjacency_matrix_1 = [
    [0, 1, 1, 1],
    [1, 0, 1, 1],
    [1, 1, 0, 1],
    [1, 1, 1, 0],
]
assert can_color_graph(adjacency_matrix_1, 4)

#58
def find_element(arr, element, start, end):

    if start == end:
        return

    mid = start + ((end - start) // 2)

    if arr[mid] == element:
        return mid
    elif arr[mid] > element:
        if arr[start] >= element:
            return find_element(arr, element, start, mid)
        else:
            return find_element(arr, element, mid, end)
    elif arr[mid] < element:
        if arr[start] <= element:
            return find_element(arr, element, start, mid)
        else:
            return find_element(arr, element, mid, end)


def find_element_main(arr, element):
    element_pos = find_element(arr, element, 0, len(arr))
    return element_pos


assert find_element_main([13, 18, 25, 2, 8, 10], 2) == 3
assert find_element_main([13, 18, 25, 2, 8, 10], 8) == 4
assert find_element_main([25, 2, 8, 10, 13, 18], 8) == 2
assert find_element_main([8, 10, 13, 18, 25, 2], 8) == 0

#62
def matrix_traversal_helper(row_count, col_count, curr_row, curr_col):

    if curr_row == row_count - 1 and curr_col == col_count - 1:
        return 1

    count = 0
    if curr_row < row_count - 1:
        count += matrix_traversal_helper(row_count, col_count,
                                         curr_row + 1, curr_col)
    if curr_col < col_count - 1:
        count += matrix_traversal_helper(row_count, col_count,
                                         curr_row, curr_col + 1)

    return count


def get_matrix_traversals(row_count, col_count):
    if not row_count or not col_count:
        return None
    count = matrix_traversal_helper(row_count, col_count, 0, 0)
    return count


assert not get_matrix_traversals(1, 0)
assert get_matrix_traversals(1, 1) == 1
assert get_matrix_traversals(2, 2) == 2
assert get_matrix_traversals(5, 5) == 70

#69
import sys


def get_pairwise_products(arr):
    pairwise_products = list()
    for i in range(len(arr)):
        for j in range(len(arr)):
            if i != j:
                pairwise_products.append([set([i, j]), arr[i] * arr[j]])

    return pairwise_products


def get_largest_product(arr):
    pairwise_products = get_pairwise_products(arr)
    max_triple = -1 * sys.maxsize
    for i in range(len(arr)):
        for prev_indices, product in pairwise_products:
            if i not in prev_indices:
                triple_prod = arr[i] * product
                if triple_prod > max_triple:
                    max_triple = triple_prod

    return max_triple


assert get_largest_product([-10, -10, 5, 2]) == 500
assert get_largest_product([-10, 10, 5, 2]) == 100

#70
def get_perfect_number(n):
    tmp_sum = 0
    for char in str(n):
        tmp_sum += int(char)

    return (n * 10) + (10 - tmp_sum)


assert get_perfect_number(1) == 19
assert get_perfect_number(2) == 28
assert get_perfect_number(3) == 37
assert get_perfect_number(10) == 109
assert get_perfect_number(11) == 118
assert get_perfect_number(19) == 190

#74
def get_mult_count(n, x):
    if n == 1:
        return n

    tuples = list()
    for i in range(1, (x + 1) // 2):
        if not x % i:
            tuples.append((i, x // i))

    return len(tuples)


assert get_mult_count(1, 1) == 1
assert get_mult_count(6, 12) == 4
assert get_mult_count(2, 4) == 1
assert get_mult_count(3, 6) == 2

#79
def can_edit(arr):
    decr_pairs = 0
    for i in range(1, len(arr)):
        if arr[i] < arr[i - 1]:
            decr_pairs += 1

    return decr_pairs <= 1


assert can_edit([10, 5, 7])
assert not can_edit([10, 5, 1])
assert can_edit([1, 10, 5, 7])

#82
class FileProxy:
    def __init__(self, contents):
        self.contents = contents
        self.offset = 0
        self.buffer = ""

    def read_7(self):
        start = self.offset
        end = min(self.offset + 7, len(self.contents))
        self.offset = end
        return self.contents[start:end].strip()

    def read_n(self, n):
        while len(self.buffer) < n:
            additional_chars = self.read_7()
            if not (additional_chars):
                break
            self.buffer += additional_chars

        n_chars = self.buffer[:n]
        self.buffer = self.buffer[n:]
        return n_chars.strip()


fp = FileProxy("Hello world")
assert fp.read_7() == "Hello w"
assert fp.read_7() == "orld"
assert fp.read_7() == ""

fp = FileProxy("Hello world")
assert fp.read_n(8) == "Hello wo"
assert fp.read_n(8) == "rld"

fp = FileProxy("Hello world")
assert fp.read_n(4) == "Hell"
assert fp.read_n(4) == "o wo"
assert fp.read_n(4) == "rld"

#85
def get_num(x, y, b):
    return x * b + y * abs(b-1)

#88
def divide(dividend, divisor):
    if not divisor:
        return

    current_sum = 0
    quotient = 0
    while current_sum <= dividend:
        quotient += 1
        current_sum += divisor

    return quotient - 1

#89
import sys


class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


def is_valid_bst_node_helper(node, lb, ub):

    if node and node.val <= ub and node.val >= lb:
        return is_valid_bst_node_helper(node.left, lb, node.val) and \
            is_valid_bst_node_helper(node.right, node.val, ub)

    return not node  # if node is None, it's a valid BST


def is_valid_bst(root):
    return is_valid_bst_node_helper(root, -sys.maxsize, sys.maxsize)



#96
def get_permutations(arr):
    if len(arr) < 2:
        return [arr]

    permutations = list()
    for i, num in enumerate(arr):
        arr_cp = arr[:i] + arr[i+1:]
        child_perms = get_permutations(arr_cp)
        for perm in child_perms:
            permutations.append([num] + perm)

    return permutations
#97
import bisect


class TimedMap:
    def __init__(self):
        self.map = dict()

    def __repr__(self):
        return str(self.map)

    def setval(self, key, value, time):
        if key not in self.map:
            self.map[key] = ([time], [value])
            return

        times, values = self.map[key]
        insertion_point = bisect.bisect(times, time)
        times.insert(insertion_point, time)
        values.insert(insertion_point, value)

    def getval(self, key, time):
        if key not in self.map:
            return

        times, values = self.map[key]
        insertion_point = bisect.bisect(times, time)
        if not insertion_point:
            return

        return values[insertion_point - 1]


d = TimedMap()
d.setval(1, 1, 0)
d.setval(1, 2, 2)
assert d.getval(1, 1) == 1
assert d.getval(1, 3) == 2

d = TimedMap()
d.setval(1, 1, 5)
assert not d.getval(1, 0)
assert d.getval(1, 10) == 1

d = TimedMap()
d.setval(1, 1, 0)
d.setval(1, 2, 0)
assert d.getval(1, 0) == 2




#101
def is_prime(num, primes):
    for prime in primes:
        if prime == num:
            return True
        if not num % prime:
            return False
    return True


def get_primes(num):
    limit = (num // 2) + 1

    candidates = list()
    primes = list()
    for i in range(2, limit):
        if is_prime(i, primes):
            primes.append(i)
            candidates.append((i, num - i))

    new_candidates = list()
    for first, second in candidates[::-1]:
        if is_prime(second, primes):
            primes.append(second)
            new_candidates.append((first, second))

    return new_candidates[-1]


assert get_primes(4) == (2, 2)
assert get_primes(10) == (3, 7)
assert get_primes(100) == (3, 97)

#102
def get_cont_arr(arr, target):
    summed = 0

    start, end = 0, 0
    i = 0
    while i < len(arr):
        if summed == target:
            return arr[start:end]
        elif summed > target:
            summed -= arr[start]
            start += 1
        else:
            summed += arr[i]
            end = i + 1
            i += 1


assert get_cont_arr([1, 2, 3, 4, 5], 0) == []
assert get_cont_arr([1, 2, 3, 4, 5], 1) == [1]
assert get_cont_arr([1, 2, 3, 4, 5], 5) == [2, 3]
assert get_cont_arr([5, 4, 3, 4, 5], 12) == [5, 4, 3]
assert get_cont_arr([5, 4, 3, 4, 5], 11) == [4, 3, 4]
assert get_cont_arr([1, 2, 3, 4, 5], 9) == [2, 3, 4]
assert get_cont_arr([1, 2, 3, 4, 5], 3) == [1, 2]

#109
# 85 is the odd-bit filter '01010101'


def swap_bits(num):
    return ((num & 85) << 1) | ((num & (85 << 1)) >> 1)

#124
from math import log, ceil


def get_num_expected(coin_tosses):
    return ceil(log(coin_tosses, 2))

#129
TOLERANCE = 10 ** -6


def almost_equal(first, second):
    # check equality with some acceptable error tolerance
    return \
        second > first - TOLERANCE and \
        second < first + TOLERANCE


def get_sqrt_helper(n, start, end):
    mid = start + ((end - start) / 2)

    if almost_equal(mid * mid, n):
        return mid
    elif mid * mid > n:
        return get_sqrt_helper(n, start, mid)
    else:
        return get_sqrt_helper(n, mid, end)


def get_sqrt(n):
    return get_sqrt_helper(n, 0, n)

#140
def get_singles(arr):
    xored = arr[0]
    for num in arr[1:]:
        xored ^= num
    x, y = 0, 0

    rightmost_set_bit = (xored & ~(xored - 1))
    for num in arr:
        if num & rightmost_set_bit:
            x ^= num
        else:
            y ^= num

    return (x, y)


# Tests

get_singles([2, 4, 6, 8, 10, 2, 6, 10]) == (4, 8)
get_singles([2, 4, 8, 8, 10, 2, 6, 10]) == (4, 6)

#148
def get_gray_code(n):
    """
    n: bits 
    """
    if n == 0:
        return ['']

    lower_grey_codes = get_gray_code(n - 1)
    l0 = ['0' + x for x in lower_grey_codes]
    l1 = ['1' + x for x in reversed(lower_grey_codes)]

    return l0 + l1

#153
def get_smallest_dist(text, w1, w2):
    dist = None
    ls_word, ls_index = None, None
    for index, word in enumerate(text.split()):
        if word == w1 or word == w2:
            if (word == w1 and ls_word == w2) or \
                    (word == w2 and ls_word == w1):
                dist = index - ls_index - 1
            ls_word = word
            ls_index = index

    return dist


# Tests
assert not get_smallest_dist(
    "hello", "hello", "world")
assert get_smallest_dist(
    "hello world", "hello", "world") == 0
assert get_smallest_dist(
    "dog cat hello cat dog dog hello cat world", "hello", "world") == 1
assert get_smallest_dist(
    "dog cat hello cat dog dog hello cat world", "dog", "world") == 2

#157
def check_palindrome_rearrangement(string):
    chars = set()
    for char in string:
        if char not in chars:
            chars.add(char)
        else:
            chars.remove(char)

    return len(chars) < 2

#161
def reverse_bits(num):
    inverted = list()
    for char in num:
        if char == '0':
            inverted.append('1')
        else:
            inverted.append('0')

    return "".join(inverted)


# Tests
assert reverse_bits('101') == '010'
assert reverse_bits('11110000111100001111000011110000') == \
    '00001111000011110000111100001111'

#172
from itertools import permutations


def get_indices(s, words):
    perms = list(permutations(words))
    perms = [x + y for (x, y) in perms]

    indices = [s.find(x) for x in perms]
    indices = [x for x in indices if x >= 0]

    return sorted(indices)

#173
def is_dict(var):
    return str(type(var)) == "<class 'dict'>"


def flatten_helper(d, flat_d, path):
    if not is_dict(d):
        flat_d[path] = d
        return

    for key in d:
        new_keypath = "{}.{}".format(path, key) if path else key
        flatten_helper(d[key], flat_d, new_keypath)


def flatten(d):
    flat_d = dict()
    flatten_helper(d, flat_d, "")
    return flat_d


# Tests

d = {
    "key": 3,
    "foo": {
        "a": 5,
        "bar": {
            "baz": 8
        }
    }
}

assert flatten(d) == {
    "key": 3,
    "foo.a": 5,
    "foo.bar.baz": 8
}

#176
def is_char_mapped(str_a, str_b):
    if len(str_a) != len(str_b):
        return False

    char_map = dict()
    for char_a, char_b in zip(str_a, str_b):
        if char_a not in char_map:
            char_map[char_a] = char_b
        elif char_map[char_a] != char_b:
            return False

    return True


# Tests
assert is_char_mapped("abc", "bcd")
assert not is_char_mapped("foo", "bar")

#194
def get_intersections(parr, qarr):
    segments = list(zip(parr, qarr))

    count = 0
    for i in range(len(segments)):
        for k in range(i):
            p1, p2 = segments[i], segments[k]
            if (p1[0] < p2[0] and p1[1] > p2[1]) or \
                    (p1[0] > p2[0] and p1[1] < p2[1]):
                count += 1

    return count


# Tests
assert get_intersections([1, 4, 5], [4, 2, 3]) == 2
assert get_intersections([1, 4, 5], [2, 3, 4]) == 0

#202
DEC_FACT = 10


def is_palindrome(num, size):
    if size == 0 or size == 1:
        return True

    fdig_factor = DEC_FACT ** (size - 1)
    fdig = num // fdig_factor
    ldig = num % DEC_FACT

    if fdig != ldig:
        return False

    new_num = (num - (fdig * fdig_factor)) // DEC_FACT
    return is_palindrome(new_num, size - 2)


def is_palindrome_helper(num):
    size = 0
    num_cp = num
    while num_cp:
        num_cp = num_cp // DEC_FACT
        size += 1

    return is_palindrome(num, size)



# Tests
assert is_palindrome_helper(121)
assert is_palindrome_helper(888)
assert not is_palindrome_helper(678)
assert not is_palindrome_helper(1678)
assert is_palindrome_helper(1661)

#203
def get_smallest(arr, start, end):
    mid = start + ((end - start) // 2)

    if arr[start] <= arr[mid]:
        if arr[end] < arr[mid]:
            return get_smallest(arr, mid + 1, end)
        else:
            return arr[start]
    elif arr[start] >= arr[mid]:
        if arr[end] > arr[mid]:
            return get_smallest(arr, start, end)
        else:
            return arr[end]


def get_smallest_helper(arr):
    smallest = get_smallest(arr, 0, len(arr) - 1)
    return smallest


# Tests
assert get_smallest_helper([5, 7, 10, 3, 4]) == 3
assert get_smallest_helper([4, 5, 7, 10, 3]) == 3
assert get_smallest_helper([3, 4, 5, 7, 10]) == 3

#211
def get_occurrences(string, pattern):
    sl, pl = len(string), len(pattern)
    occurrences = list()

    for i in range(sl - pl + 1):
        if string[i:i+pl] == pattern:
            occurrences.append(i)

    return occurrences

#218
class Node:
    def __init__(self, iden):
        self.iden = iden

    def __hash__(self):
        return hash(self.iden)

    def __eq__(self, other):
        return self.iden == other.iden

    def __repr__(self):
        return str(self.iden)


class Edge:
    def __init__(self, src, tgt):
        self.src = src
        self.tgt = tgt

    def __hash__(self):
        return hash((self.src, self.tgt))

    def __eq__(self, other):
        return self.src == other.src and self.tgt == other.tgt

    def __repr__(self):
        return "{}->{}".format(self.src, self.tgt)

    def reverse(self):
        tmp_node = self.src
        self.src = self.tgt
        self.tgt = tmp_node


class Graph:
    def __init__(self):
        self.nodes = set()
        self.edges = set()

    def add_node(self, node):
        if node in self.nodes:
            return
        self.nodes.add(node)

    def add_edge(self, src_node, tgt_node):
        self.edges.add(Edge(src_node, tgt_node))

    def reverse_edges(self):
        self.edges = [Edge(x.tgt, x.src) for x in self.edges]

    def get_edges(self):
        return self.edges


# Tests
g = Graph()
a = Node('a')
b = Node('b')
c = Node('c')

g.add_node(a)
g.add_node(b)
g.add_node(c)
g.add_edge(a, b)
g.add_edge(b, c)
edges = g.get_edges()
assert Edge(a, b) in edges and Edge(b, c) in edges and len(edges) == 2

g.reverse_edges()
edges = g.get_edges()
assert Edge(b, a) in edges and Edge(c, b) in edges and len(edges) == 2

#221
def get_nth_sevenish(n):
    if n < 1:
        raise Exception("Invalid value for 'n'")

    power = 0
    sevenish_nums = list()
    while len(sevenish_nums) < n:
        num = 7 ** power
        new_sevenish_nums = [num]
        for old in sevenish_nums:
            if len(sevenish_nums) + len(new_sevenish_nums) == n:
                return new_sevenish_nums[-1]
            new_sevenish_nums.append(num + old)

        sevenish_nums += new_sevenish_nums
        power += 1

    return sevenish_nums[-1]


# Tests
assert get_nth_sevenish(1) == 1
assert get_nth_sevenish(2) == 7
assert get_nth_sevenish(3) == 8
assert get_nth_sevenish(10) == 350

#243
import sys


def split(arr, k):
    if k == 1:
        return ([arr], sum(arr))

    min_val = sys.maxsize
    min_cand = None
    for i in range(len(arr)):
        arr_1, sum_1 = ([arr[:i]], sum(arr[:i]))
        arr_2, sum_2 = split(arr[i:], k - 1)
        candidate = arr_1 + arr_2, max(sum_1, sum_2)
        if candidate[1] < min_val:
            min_val = candidate[1]
            min_cand = candidate

    return min_cand


def split_helper(arr, k):
    return split(arr, k)[1]


# Tests
assert split_helper([5, 1, 2, 7, 3, 4], 3) == 8

#266
ALPHA_SIZE = 26
APLHA_ASCII_OFFSET = 65


class WordCode:
    def __init__(self, word):
        self.word = word
        self.vec = [0 for _ in range(ALPHA_SIZE)]
        for ch in word:
            ind = ord(ch) - APLHA_ASCII_OFFSET
            self.vec[ind] += 1

    def __repr__(self):
        return "{}=>{}".format(self.word, self.vec)

    def __sub__(self, other):
        result = list()
        for i in range(ALPHA_SIZE):
            result.append(max(0, self.vec[i] - other.vec[i]))

        return result


def get_step_words(word, dictionary):
    step_words = set()
    wc = WordCode(word)
    for dword in dictionary:
        dwc = WordCode(dword)
        diff = dwc - wc
        if sum(diff) == 1:
            step_words.add(dword)

    return step_words


# Tests
assert get_step_words("APPLE", {"APPEAL"}) == {"APPEAL"}
assert get_step_words("APPLE", {"APPEAL", "APPLICT"}) == {"APPEAL"}
assert get_step_words("APPLE", {"APPEAL", "APPLICT", "APPLES"}) == {"APPEAL", "APPLES"}


#282
def contains_pytrip(arr):
    squared = [x * x for x in arr]
    set_of_squares = set(squared)
    for i in range(len(squared) - 1):
        for k in range(i + 1, len(squared) - 1):
            summed = squared[i] + squared[k]
            if summed in set_of_squares:
                return True

    return False


# Tests
assert contains_pytrip([3, 4, 5, 6, 7])
assert not contains_pytrip([3, 5, 6, 7])

#303
from math import isclose

FLOAT_EQUALITY_TOLERANCE = 0.5


def get_angle_for_hour(hour: int, minute: int):
    minute_offset = minute / 12
    hour_angle = (hour * 30) + minute_offset
    return hour_angle


def get_angle_for_minute(minute: int):
    return minute * 6


def get_angle(hhmm_time: str):
    hour, minute = map(int, hhmm_time.split(":"))
    hour %= 12
    ha = get_angle_for_hour(hour, minute)
    ma = get_angle_for_minute(minute)

    angle = abs(ha - ma)
    return angle if angle < 180 else 360 - angle


# Tests
assert isclose(get_angle("12:20"), 118, abs_tol=FLOAT_EQUALITY_TOLERANCE)
assert isclose(get_angle("12:00"), 0, abs_tol=FLOAT_EQUALITY_TOLERANCE)
assert isclose(get_angle("6:30"), 3, abs_tol=FLOAT_EQUALITY_TOLERANCE)
assert isclose(get_angle("3:45"), 176, abs_tol=FLOAT_EQUALITY_TOLERANCE)

#310
def get_set_bits(num):
    if not num:
        return 0

    max_pow, max_pow_of_two = 0, 1
    while max_pow_of_two - 1 <= num:
        max_pow_of_two *= 2
        max_pow += 1
    max_pow_of_two //= 2
    max_pow -= 1

    remainder = num - (max_pow_of_two - 1)
    set_bits = ((max_pow * max_pow_of_two) // 2)

    set_bits = set_bits + get_set_bits(remainder)

    return set_bits


# Tests
assert get_set_bits(0) == 0
assert get_set_bits(1) == 1
assert get_set_bits(2) == 2
assert get_set_bits(3) == 4
assert get_set_bits(4) == 5


#311
def find_peak(arr):
    if not arr:
        return None

    mid = len(arr) // 2

    if mid > 0 and arr[mid] > arr[mid - 1] and \
            mid < len(arr) and arr[mid] > arr[mid + 1]:
        return arr[mid]

    if mid > 0 and arr[mid] > arr[mid - 1]:
        return find_peak(arr[:mid])

    return find_peak(arr[mid + 1:])


# Tests
assert find_peak([0, 2, 4, 5, 3, 1]) == 5


#317
# if there are 2 consecutive numbers, the least significant
# bit will be 0 once, which means the result of an AND on
# the last bit will be zero
from math import log2, ceil


def bitwise_and_slow(start, end):
    result = end
    for num in range(start, end):
        result &= num

    return result


def bitwise_and(start, end):
    diff = end - start + 1
    power_diff = ceil(log2(diff))
    power_end = ceil(log2(end))

    result = start & (2**power_end - 2**power_diff)
    return result


# Tests
assert bitwise_and_slow(5, 6) == 4
assert bitwise_and(5, 6) == 4
assert bitwise_and_slow(126, 127) == 126
assert bitwise_and(126, 127) == 126
assert bitwise_and_slow(129, 215) == 128
assert bitwise_and(193, 215) == 192

#324
import sys


def get_min_steps(mice, holes, largest_step=-sys.maxsize):
    if not mice:
        return largest_step

    mouse = mice[0]
    min_steps = list()
    for hole in holes:
        diff = abs(mouse - hole)
        min_steps.append(
            get_min_steps(mice[1:], holes - {hole}, max(largest_step, diff))
        )

    return min(min_steps)


# Tests
assert get_min_steps(mice=[1, 4, 9, 15], holes={10, -5, 0, 16}) == 6

#332
def get_variables(m, n):
    candidates = list()
    for a in range(m // 2 + 1):
        b = m - a
        if a ^ b == n:
            candidates.append((a, b))

    return candidates


# Tests
assert get_variables(100, 4) == [(48, 52)]

#334
OPERATORS = {'+', '-', '*', '/'}
TARGET = 24


def possible(arr):
    if len(arr) == 1:
        return arr[0] == TARGET

    new_possibilities = list()
    for si in range(len(arr) - 1):
        for operator in OPERATORS:
            num_1 = arr[si]
            num_2 = arr[si + 1]
            try:
                possibility = \
                    arr[:si] + \
                    [eval("{}{}{}".format(num_1, operator, num_2))] + \
                    arr[si + 2:]
                new_possibilities.append(possibility)
            except Exception:
                pass

    return any([possible(x) for x in new_possibilities])


# Tests
assert possible([5, 2, 7, 8])
assert not possible([10, 10, 10, 10])

#336
class Node:
    def __init__(self, val):
        self.val = val
        self.l, self.r = None, None


def get_distinct_ways(node):
    if node and node.l and node.r:
        return 2 * get_distinct_ways(node.l) * get_distinct_ways(node.r)

    return 1


# Tests
a = Node(3)
b = Node(2)
c = Node(1)
a.l = b
a.r = c
assert get_distinct_ways(a) == 2

#339
def get_twos_sum(result, arr):
    i, k = 0, len(arr) - 1
    while i < k:
        a, b = arr[i], arr[k]
        res = a + b
        if res == result:
            return (a, b)
        elif res < result:
            i += 1
        else:
            k -= 1


def get_threes_sum(result, arr):
    arr.sort()
    for i in range(len(arr)):
        c = arr[i]
        if c > result:
            continue
        twos = get_twos_sum(result - c, arr[:i] + arr[i+1:])
        if twos:
            return True

    return get_twos_sum(result, arr)


# Tests
assert get_threes_sum(49, [20, 303, 3, 4, 25])
assert not get_threes_sum(50, [20, 303, 3, 4, 25])

#338
def get_ones(num: int):
    binary = str(bin(num))
    count = 0
    for ch in binary:
        if ch == '1':
            count += 1

    return count


def get_next(num: int):
    inc = 1
    base_count = get_ones(num)
    while True:
        next_num = num + inc
        new_count = get_ones(next_num)
        if base_count == new_count:
            return next_num
        inc += 1


# Tests
assert get_next(6) == 9

#353
def get_max_hist_area(arr, start, end):
    if start == end:
        return 0

    curr_area = (end - start) * min(arr[start:end])
    opt_1 = get_max_hist_area(arr, start, end - 1)
    opt_2 = get_max_hist_area(arr, start + 1, end)

    return max(curr_area, opt_1, opt_2)


def get_max_hist_area_helper(arr):
    return get_max_hist_area(arr, 0, len(arr))


# Tests
assert get_max_hist_area_helper([1, 3, 2, 5]) == 6













































































