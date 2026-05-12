"""
FoodSeg103 class names and label mappings.

The FoodSeg103 dataset has 104 labels (0 = background, 1-103 = food categories).
These names are used as text prompts for the MobileCLIP open-vocabulary branch
and for display/evaluation purposes.
"""

# 104 classes: index 0 is background, indices 1-103 are food categories
FOODSEG103_CLASSES = [
    "background",           # 0
    "candy",                # 1
    "egg tart",             # 2
    "french fries",         # 3
    "chocolate",            # 4
    "biscuit",              # 5
    "popcorn",              # 6
    "pudding",              # 7
    "ice cream",            # 8
    "cheese butter",        # 9
    "cake",                 # 10
    "wine",                 # 11
    "milkshake",            # 12
    "coffee",               # 13
    "juice",                # 14
    "milk",                 # 15
    "tea",                  # 16
    "almond",               # 17
    "red beans",            # 18
    "cashew",               # 19
    "dried cranberries",    # 20
    "soy",                  # 21
    "walnut",               # 22
    "peanut",               # 23
    "egg",                  # 24
    "apple",                # 25
    "date",                 # 26
    "apricot",              # 27
    "avocado",              # 28
    "banana",               # 29
    "strawberry",           # 30
    "cherry",               # 31
    "blueberry",            # 32
    "raspberry",            # 33
    "mango",                # 34
    "olives",               # 35
    "peach",                # 36
    "lemon",                # 37
    "pear",                 # 38
    "fig",                  # 39
    "pineapple",            # 40
    "grape",                # 41
    "kiwi",                 # 42
    "melon",                # 43
    "orange",               # 44
    "watermelon",           # 45
    "steak",                # 46
    "pork",                 # 47
    "chicken duck",         # 48
    "sausage",              # 49
    "fried meat",           # 50
    "lamb",                 # 51
    "sauce",                # 52
    "crab",                 # 53
    "fish",                 # 54
    "shellfish",            # 55
    "shrimp",               # 56
    "soup",                 # 57
    "bread",                # 58
    "corn",                 # 59
    "hamburg",              # 60
    "pizza",                # 61
    "hanamaki baozi",       # 62
    "wonton dumplings",     # 63
    "pasta",                # 64
    "noodles",              # 65
    "rice",                 # 66
    "pie",                  # 67
    "tofu",                 # 68
    "eggplant",             # 69
    "potato",               # 70
    "garlic",               # 71
    "cauliflower",          # 72
    "tomato",               # 73
    "kelp",                 # 74
    "seaweed",              # 75
    "spring onion",         # 76
    "rape",                 # 77
    "ginger",               # 78
    "okra",                 # 79
    "lettuce",              # 80
    "pumpkin",              # 81
    "cucumber",             # 82
    "white radish",         # 83
    "carrot",               # 84
    "asparagus",            # 85
    "bamboo shoots",        # 86
    "broccoli",             # 87
    "celery stem",          # 88
    "mushroom",             # 89
    "green beans",          # 90
    "bean sprouts",         # 91
    "spinach",              # 92
    "green pepper",         # 93
    "red pepper",           # 94
    "pepper",               # 95
    "onion",                # 96
    "sweet potato",         # 97
    "beet",                 # 98
    "zucchini",             # 99
    "cabbage",              # 100
    "mixed vegetables",     # 101
    "salad",                # 102
    "other ingredients",    # 103
]

NUM_CLASSES = len(FOODSEG103_CLASSES)  # 104

# Mapping from class index to class name
ID2LABEL = {i: name for i, name in enumerate(FOODSEG103_CLASSES)}

# Mapping from class name to class index
LABEL2ID = {name: i for i, name in enumerate(FOODSEG103_CLASSES)}
