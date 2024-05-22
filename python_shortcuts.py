# python_shortcuts.py

# List of handy Python shortcuts

# 1. Swapping variables
a, b = 1, 2
a, b = b, a
print(f"Swapped values: a = {a}, b = {b}")

# 2. List comprehensions
squares = [x**2 for x in range(10)]
print(f"Squares: {squares}")

# 3. Dictionary comprehensions
square_dict = {x: x**2 for x in range(10)}
print(f"Square dict: {square_dict}")

# 4. Set comprehensions
square_set = {x**2 for x in range(10)}
print(f"Square set: {square_set}")

# 5. Inline if-else
age = 18
status = "adult" if age >= 18 else "minor"
print(f"Status: {status}")

# 6. Multiple assignment
name, age, city = "Alice", 30, "New York"
print(f"Name: {name}, Age: {age}, City: {city}")

# 7. Unpacking sequences
data = ('Alice', 30, 'New York')
name, age, city = data
print(f"Name: {name}, Age: {age}, City: {city}")

# 8. Merging dictionaries (Python 3.9+)
dict1 = {'a': 1, 'b': 2}
dict2 = {'b': 3, 'c': 4}
merged_dict = dict1 | dict2
print(f"Merged dictionary: {merged_dict}")

# 9. Using zip to iterate over multiple sequences
names = ["Alice", "Bob", "Charlie"]
ages = [30, 25, 35]
for name, age in zip(names, ages):
    print(f"{name} is {age} years old")

# 10. Enumerate for index and value
fruits = ['apple', 'banana', 'cherry']
for index, fruit in enumerate(fruits):
    print(f"Index {index}: {fruit}")

# 11. Lambda functions for short anonymous functions
add = lambda x, y: x + y
print(f"Sum: {add(5, 3)}")

# 12. Using * to unpack arguments
def multiply(x, y, z):
    return x * y * z

nums = (2, 3, 4)
print(f"Product: {multiply(*nums)}")

# 13. Using ** to unpack keyword arguments
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

params = {"name": "Alice", "greeting": "Hi"}
print(greet(**params))

# 14. List slicing
lst = [1, 2, 3, 4, 5]
print(f"Sliced list (1:3): {lst[1:3]}")
print(f"Sliced list (start to 3): {lst[:3]}")
print(f"Sliced list (3 to end): {lst[3:]}")
print(f"Reversed list: {lst[::-1]}")

# 15. Using Counter for frequency counting
from collections import Counter
words = ["apple", "banana", "apple", "orange", "banana", "apple"]
word_count = Counter(words)
print(f"Word count: {word_count}")

# 16. ChainMap for combining multiple dictionaries
from collections import ChainMap
dict1 = {'a': 1, 'b': 2}
dict2 = {'b': 3, 'c': 4}
combined = ChainMap(dict1, dict2)
print(f"Combined ChainMap: {combined}")

# 17. Named tuples for simple classes
from collections import namedtuple
Point = namedtuple('Point', 'x y')
p = Point(11, 22)
print(f"Point: {p}, x: {p.x}, y: {p.y}")

# 18. Using itertools for efficient looping
from itertools import permutations, combinations
print(f"Permutations of 'AB': {list(permutations('AB'))}")
print(f"Combinations of 'AB': {list(combinations('AB', 1))}")

# 19. Context managers for resource management
with open('example.txt', 'w') as f:
    f.write("Hello, world!")

# 20. f-strings for formatted strings (Python 3.6+)
name = "Alice"
age = 30
print(f"My name is {name} and I am {age} years old.")

# 21. Ternary conditional operator
result = "Even" if age % 2 == 0 else "Odd"
print(f"Age is {result}")

# 22. Flatten a list of lists
nested_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flat_list = [item for sublist in nested_list for item in sublist]
print(f"Flat list: {flat_list}")

# 23. Transpose a matrix
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
transposed = list(zip(*matrix))
print(f"Transposed matrix: {transposed}")

# 24. Using defaultdict for missing keys
from collections import defaultdict
dd = defaultdict(int)
dd['key'] += 1
print(f"defaultdict: {dd}")

# 25. Using get() with dictionaries
d = {'a': 1, 'b': 2}
value = d.get('c', 0)
print(f"Value for key 'c': {value}")

# 26. Using slice object
data = [0, 1, 2, 3, 4, 5, 6]
s = slice(1, 5, 2)
print(f"Sliced data: {data[s]}")

# 27. Using any() and all()
conditions = [True, False, True]
print(f"Any true: {any(conditions)}")
print(f"All true: {all(conditions)}")

# 28. Using enumerate to create a dict
elements = ['a', 'b', 'c']
enum_dict = dict(enumerate(elements))
print(f"Enumerate dict: {enum_dict}")

# 29. String join method
words = ['Hello', 'world']
sentence = ' '.join(words)
print(f"Sentence: {sentence}")

# 30. Reverse a string
reversed_str = 'hello'[::-1]
print(f"Reversed string: {reversed_str}")

# 31. Check for palindrome
word = "racecar"
is_palindrome = word == word[::-1]
print(f"Is palindrome: {is_palindrome}")

# 32. Using filter with lambda
nums = [1, 2, 3, 4, 5, 6]
even_nums = list(filter(lambda x: x % 2 == 0, nums))
print(f"Even numbers: {even_nums}")

# 33. Using map with lambda
squared_nums = list(map(lambda x: x ** 2, nums))
print(f"Squared numbers: {squared_nums}")

# 34. Using reduce for cumulative operation
from functools import reduce
product = reduce(lambda x, y: x * y, nums)
print(f"Product of numbers: {product}")

# 35. Using sorted with custom key
words = ["apple", "banana", "cherry"]
sorted_words = sorted(words, key=lambda x: len(x))
print(f"Sorted by length: {sorted_words}")

# 36. Checking for subset and superset
set_a = {1, 2, 3}
set_b = {1, 2}
is_subset = set_b.issubset(set_a)
is_superset = set_a.issuperset(set_b)
print(f"Is subset: {is_subset}, Is superset: {is_superset}")

# 37. Using itertools.chain for flattening lists
from itertools import chain
nested = [[1, 2, 3], [4, 5], [6, 7, 8]]
flat = list(chain(*nested))
print(f"Flattened list: {flat}")

# 38. Using a generator expression
gen_exp = (x**2 for x in range(10))
print(f"Generator expression: {list(gen_exp)}")

# 39. Reading a file line by line
with open('example.txt', 'r') as f:
    lines = f.readlines()
print(f"File lines: {lines}")

# 40. Writing multiple lines to a file
lines_to_write = ["Line 1", "Line 2", "Line 3"]
with open('example.txt', 'w') as f:
    f.writelines('\n'.join(lines_to_write))

# 41. Simple class with __str__ method
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __str__(self):
        return f"{self.name}, {self.age} years old"

person = Person("Alice", 30)
print(person)

# 42. Using dataclasses (Python 3.7+)
from dataclasses import dataclass

@dataclass
class Car:
    make: str
    model: str
    year: int

car = Car(make="Toyota", model="Corolla", year=2020)
print(car)

# 43. Using __slots__ to save memory
class PointWithSlots:
    __slots__ = ['x', 'y']

    def __init__(self, x, y):
        self.x = x
        self.y = y

p = PointWithSlots(10, 20)
print(f"Point with slots: ({p.x}, {p.y})")

# 44. Using @property decorator
class Circle:
    def __init__(self, radius):
        self._radius = radius

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        if value >= 0:
            self._radius = value
        else:
            raise ValueError("Radius must be non-negative")

circle = Circle(5)
print(f"Circle radius: {circle.radius}")
circle.radius = 10
print(f"Updated circle radius: {circle.radius}")

# 45. Using collections.deque for fast appends and pops
from collections import deque
dq = deque([1, 2, 3])
dq.appendleft(0)
dq.append(4)
print(f"Deque: {dq}")

# 46. Using bisect to maintain a sorted list
import bisect
sorted_list = [1, 2, 4, 5]
bisect.insort(sorted_list, 3)
print(f"Sorted list with insort: {sorted_list}")

# 47. Using heapq for a priority queue
import heapq
heap = [3, 1, 4, 1, 5, 9]
heapq.heapify(heap)
print(f"Heapified list: {heap}")
heapq.heappush(heap, 2)
print(f"Heap after push: {heap}")
smallest = heapq.heappop(heap)
print(f"Smallest element: {smallest}")

# 48. Using lru_cache for memoization
from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(f"Fibonacci(10): {fibonacci(10)}")

# 49. Using partial to fix arguments
from functools import partial

def power(base, exponent):
    return base ** exponent

square = partial(power, exponent=2)
print(f"Square of 4: {square(4)}")

# 50. Using namedtuple for structured data
from collections import namedtuple

Person = namedtuple('Person', 'name age')
alice = Person(name="Alice", age=30)
print(f"Namedtuple person: {alice}")

# 51. Using frozenset for immutable sets
immutable_set = frozenset([1, 2, 3])
print(f"Frozenset: {immutable_set}")

# 52. Using islice for slicing iterators
from itertools import islice

iterable = iter(range(10))
sliced = list(islice(iterable, 2, 6))
print(f"Sliced iterator: {sliced}")

# 53. Using groupby to group elements
from itertools import groupby

data = sorted([('a', 1), ('b', 2), ('a', 3)])
grouped = {k: list(v) for k, v in groupby(data, key=lambda x: x[0])}
print(f"Grouped data: {grouped}")

# 54. Using compress to filter by a mask
from itertools import compress

data = range(10)
selectors = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
filtered = list(compress(data, selectors))
print(f"Filtered by mask: {filtered}")

# 55. Using combinations_with_replacement
from itertools import combinations_with_replacement

comb_wr = list(combinations_with_replacement('ABC', 2))
print(f"Combinations with replacement: {comb_wr}")

# 56. Simple recursive function
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n-1)

print(f"Factorial(5): {factorial(5)}")

# 57. Using contextlib for simple context managers
from contextlib import contextmanager

@contextmanager
def simple_context():
    print("Entering")
    yield
    print("Exiting")

with simple_context():
    print("Inside context")

# 58. Using counter with most_common
from collections import Counter

counter = Counter("abracadabra")
print(f"Most common elements: {counter.most_common(3)}")

# 59. Using defaultdict for list
d = defaultdict(list)
d['key'].append('value')
print(f"defaultdict with list: {d}")

# 60. Using tarfile to compress files
import tarfile

with tarfile.open('sample.tar.gz', 'w:gz') as tar:
    tar.add('example.txt')

# 61. Using zipfile to compress files
import zipfile

with zipfile.ZipFile('sample.zip', 'w') as zipf:
    zipf.write('example.txt')

# 62. Using pathlib for file paths
from pathlib import Path

path = Path('example.txt')
print(f"File name: {path.name}")
print(f"File suffix: {path.suffix}")
print(f"File stem: {path.stem}")

# 63. Using tempfile for temporary files
import tempfile

with tempfile.TemporaryFile() as tempf:
    tempf.write(b'Some data')
    tempf.seek(0)
    print(f"Temporary file content: {tempf.read()}")

# 64. Using Enum for enumerations
from enum import Enum

class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3

print(f"Color.RED: {Color.RED}")

# 65. Using type hints for function annotations
def greet(name: str) -> str:
    return f"Hello, {name}"

print(greet("Alice"))

# 66. Using dataclasses for boilerplate code
from dataclasses import dataclass

@dataclass
class Point:
    x: int
    y: int

point = Point(10, 20)
print(f"Dataclass point: {point}")

# 67. Using __repr__ for object representation
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Point({self.x}, {self.y})"

p = Point(10, 20)
print(p)

# 68. Using __str__ for readable object representation
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __str__(self):
        return f"{self.name}, {self.age} years old"

print(Person("Alice", 30))

# 69. Using slots to save memory
class Point:
    __slots__ = ['x', 'y']

    def __init__(self, x, y):
        self.x = x
        self.y = y

p = Point(10, 20)
print(p.x, p.y)

# 70. Using generators for memory efficiency
def generate_numbers(n):
    for i in range(n):
        yield i

gen = generate_numbers(10)
print(list(gen))

# 71. Using asyncio for asynchronous programming
import asyncio

async def say_hello():
    print("Hello")
    await asyncio.sleep(1)
    print("World")

asyncio.run(say_hello())

# 72. Using dataclass for simple data containers
from dataclasses import dataclass

@dataclass
class Point:
    x: int
    y: int

p = Point(10, 20)
print(p)

# 73. Using collections.Counter for counting
from collections import Counter

words = ["apple", "banana", "apple"]
counter = Counter(words)
print(counter)

# 74. Using itertools.chain for flattening
from itertools import chain

lists = [[1, 2, 3], [4, 5], [6, 7, 8]]
flat = list(chain.from_iterable(lists))
print(flat)

# 75. Using partial for function customization
from functools import partial

def power(base, exponent):
    return base ** exponent

square = partial(power, exponent=2)
print(square(4))

# 76. Using map for element-wise operations
numbers = [1, 2, 3]
squares = list(map(lambda x: x**2, numbers))
print(squares)

# 77. Using filter for filtering elements
numbers = [1, 2, 3, 4, 5, 6]
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(evens)

# 78. Using reduce for aggregation
from functools import reduce

numbers = [1, 2, 3, 4, 5]
total = reduce(lambda x, y: x + y, numbers)
print(total)

# 79. Using namedtuple for structured data
from collections import namedtuple

Point = namedtuple('Point', ['x', 'y'])
p = Point(10, 20)
print(p)

# 80. Using frozenset for immutable sets
fs = frozenset([1, 2, 3])
print(fs)

# 81. Using heapq for priority queue
import heapq

numbers = [5, 7, 9, 1, 3]
heapq.heapify(numbers)
print(numbers)

# 82. Using bisect for binary search
import bisect

numbers = [1, 2, 4, 5]
bisect.insort(numbers, 3)
print(numbers)

# 83. Using itertools.islice for slicing iterators
from itertools import islice

iterable = range(10)
sliced = list(islice(iterable, 2, 6))
print(sliced)

# 84. Using zip_longest for padding
from itertools import zip_longest

a = [1, 2]
b = [3, 4, 5]
zipped = list(zip_longest(a, b, fillvalue=0))
print(zipped)

# 85. Using permutations for permutations
from itertools import permutations

perm = list(permutations('ABC'))
print(perm)

# 86. Using combinations for combinations
from itertools import combinations

comb = list(combinations('ABC', 2))
print(comb)

# 87. Using combinations_with_replacement
from itertools import combinations_with_replacement

comb_wr = list(combinations_with_replacement('ABC', 2))
print(comb_wr)

# 88. Using lru_cache for memoization
from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(10))

# 89. Using contextlib for context managers
from contextlib import contextmanager

@contextmanager
def simple_context():
    print("Entering")
    yield
    print("Exiting")

with simple_context():
    print("Inside")

# 90. Using deque for fast appends/pops
from collections import deque

dq = deque([1, 2, 3])
dq.appendleft(0)
dq.append(4)
print(dq)

# 91. Using Enum for enumerations
from enum import Enum

class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3

print(Color.RED)

# 92. Using pathlib for file paths
from pathlib import Path

path = Path('example.txt')
print(path.name, path.suffix)

# 93. Using tempfile for temp files
import tempfile

with tempfile.TemporaryFile() as tempf:
    tempf.write(b'Some data')
    tempf.seek(0)
    print(tempf.read())

# 94. Using type hints for annotations
def greet(name: str) -> str:
    return f"Hello, {name}"

print(greet("Alice"))

# 95. Using dataclass for data containers
from dataclasses import dataclass

@dataclass
class Point:
    x: int
    y: int

p = Point(10, 20)
print(p)

# 96. Using collections.ChainMap for multiple dicts
from collections import ChainMap

dict1 = {'a': 1}
dict2 = {'b': 2}
chain = ChainMap(dict1, dict2)
print(chain)

# 97. Using tarfile for compression
import tarfile

with tarfile.open('sample.tar.gz', 'w:gz') as tar:
    tar.add('example.txt')

# 98. Using zipfile for compression
import zipfile

with zipfile.ZipFile('sample.zip', 'w') as zipf:
    zipf.write('example.txt')

# 99. Using __slots__ to save memory
class Point:
    __slots__ = ['x', 'y']

    def __init__(self, x, y):
        self.x = x
        self.y = y

p = Point(10, 20)
print(p.x, p.y)

# 100. Using generators for memory efficiency
def generate_numbers(n):
    for i in range(n):
        yield i

gen = generate_numbers(10)
print(list(gen))

if __name__ == "__main__":
    print("This script contains 100 handy Python shortcuts.")
