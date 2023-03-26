import random
import zipfile

# Other download
import urllib.request
dl_file = "https://github.com/sebgur/SDev.Python/blob/main/models/readme.txt"
urllib.request.urlretrieve(dl_file, "mod2.txt")

# Download
import requests
URL = "https://instagram.com/favicon.ico"
response = requests.get(URL)
open("instagram.ico", "wb")

# Unzipping files
path_to_zip_file = "/content/models.zip"
with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
    zip_ref.extractall("model_2_assets")

# Pick randomly from a list
print("Pick randomly from a list")
original_list = ['alpha', 'beta', 'gamma', 'delta']
print("Original list: ")
print(original_list)
print("Picked list: ")
my_choices = random.choices(original_list, k=6)
print(my_choices)

# Arrays
print("Arrays")
x = [[1, 2, 3], [5, 6, 7], [11, 22, 33], [44, 55, 66]]
print(x[0][1])

fixed_size = [2] * 5
print(fixed_size)

# Dictionaries
print("Dictionaries")
alpha = [1, 2, 3, 4]
beta = ['a', 'b', 'c', 'd']
gamma = ['aa', 'bb', 'cc', 'dd']
dic = {'Alpha': alpha, 'Beta': beta, 'Gamma': gamma}

# Ternary operator
print("Ternary Operator")
b = 5
b = 1.5 if b > 6 else 12
print(b)

# Get construct
print("Get construct")
dic = {'Members': ['J', 'O', 'P'], 'Values': [5, 6, 7]}
result = dic.get('Members', [1, 2])
print(result)

# While loop
print("While Loop")
i = 5
while i < 10:
    print(i)
    i = i + 1
else:
    print("done")

# Assign several points at once
a = 1
b = 2
c = 3
sum1 = 0
for item in a, b, c:
    sum1 += item
print(sum1)

a, b, c = a + 1, b + 1, c + 1
print(a, b, c)

# Find index of maximum
e = [1.1, 0.5, 2.3]
b = ['a', 'b', 'c']
m = max(e)
print(m)
print(e.index(max(e)))
print(b[e.index(max(e))])

# Sort vector according to index of other vector
# That is, vector v1 contains unordered numbers, vector v2 has same size and contains strings,
# reorder v2 according to the increasing order of v1
v1 = [1, 3, 2, 4]
v2 = ['a', 'c', 'b', 'd']
v3 = ['e', 'g', 'f', 'h']
m = zip(v1, v2, v3)
sorted_m = sorted(m)
print(sorted_m)
sorted_v2 = [(x[1], x[2]) for x in sorted_m]
print(sorted_v2)

# Print result of zip
print(list(zip(v1, v3)))

# Print
i = 2567
x = 100000000.126
y = 0.45
print("Value of x: {a:,.2f}, y: {b:.3f}, i: {c:06d}".format(a=x, b=y, c=i))
# print('Precision: {:.02%}, Recall: {:.0.02%} [Percentile Detection]'.format(x, y))

