"""
EcoPrice - Perishable vs Non-Perishable Classification Engine

Classifies incoming delivery items as perishable or non-perishable
using a multi-signal keyword and category verification system.

Perishable items are admitted into the active EcoPrice dynamic pricing pipeline.
Non-perishables (toiletries, cleaning supplies, sanitizers, hardware, dry provisions)
are strictly rejected.

Classification Signals (checked in priority order):
  1. Product name non-perishable override (e.g. sanitizer, soap, electronics, toiletries).
  2. Category non-perishable department match (e.g. cleaning, hardware, personal care).
  3. Product name perishable food item match (e.g. chicken, fish, milk, tomato, apple).
  4. Category plausibility validation: items submitted under a specific department
     (e.g. 'Meat & Fish', 'Dairy') must match that department's perishable vocabulary.
  5. Shelf life and expiry heuristics for generic fresh departments.
"""

import re
from typing import Tuple

# ================================================================
# PERISHABLE DEPARTMENT LEXICONS
# Used for Category Plausibility Validation.
# ================================================================
CATEGORY_LEXICONS = {
    "meat & fish": {
        "chicken", "mutton", "beef", "pork", "lamb", "fish", "prawn", "prawns", "shrimp", "shrimps",
        "crab", "crabs", "lobster", "squid", "octopus", "clam", "mussel", "oyster", "salmon", "tuna",
        "pomfret", "rohu", "catla", "seer", "kingfish", "mackerel", "sardine", "anchovy", "tilapia",
        "basa", "carp", "hilsa", "trout", "cod", "keema", "kheema", "mince", "meat", "poultry", "seafood",
        "cutlet", "sausage", "salami", "bacon", "ham", "pepperoni", "kebab", "kabab", "tikka", "liver",
        "kidney", "wings", "breast", "drumstick", "thigh", "ribs", "steak", "fillet", "duck", "turkey",
        "quail", "broiler", "meat curry", "fish curry", "boneless", "curry cut"
    },
    "dairy": {
        "milk", "curd", "yogurt", "yoghurt", "paneer", "cheese", "butter", "ghee", "cream", "malai",
        "khoa", "khoya", "lassi", "buttermilk", "chaas", "dahi", "thayir", "thyr", "mozzarella",
        "cheddar", "gouda", "parmesan", "ricotta", "cottage cheese", "mascarpone", "brie", "feta",
        "milma", "amul", "nandini", "verka", "mother dairy", "milk mist", "sour cream", "condensed milk",
        "paneer tikka", "tofu"
    },
    "fruits": {
        "mango", "apple", "orange", "banana", "grape", "grapes", "pineapple", "papaya", "guava",
        "pomegranate", "watermelon", "melon", "cantaloupe", "honeydew", "kiwi", "strawberry",
        "strawberries", "blueberry", "blueberries", "raspberry", "cherry", "cherries", "plum",
        "peach", "apricot", "lychee", "litchi", "chiku", "chikoo", "sapota", "custard apple",
        "sitaphal", "jackfruit", "fig", "dragonfruit", "dragon fruit", "passion fruit", "starfruit",
        "pear", "pears", "avocado", "lemon", "lemons", "lime", "limes", "mosambi", "sweet lime",
        "grapefruit", "gooseberry", "amla", "poka", "nendran", "robusta", "njali", "coconut tender",
        "tender coconut", "elaneer", "rambutan", "mangosteen", "durian", "pomelo", "papita", "seb", "kela", "santara"
    },
    "vegetables": {
        "tomato", "potato", "onion", "garlic", "ginger", "carrot", "beetroot", "radish", "turnip",
        "cabbage", "cauliflower", "broccoli", "spinach", "palak", "methi", "fenugreek", "lettuce",
        "capsicum", "bell pepper", "chilli", "chillies", "chilly", "pepper", "cucumber", "zucchini",
        "pumpkin", "bottle gourd", "bitter gourd", "ridge gourd", "snake gourd", "ash gourd", "ivy gourd",
        "drumstick", "ladyfinger", "okra", "bhindi", "beans", "french beans", "cluster beans",
        "peas", "green peas", "matar", "corn", "sweet corn", "baby corn", "mushroom", "mushrooms",
        "spring onion", "leek", "celery", "asparagus", "artichoke", "coriander", "cilantro",
        "curry leaves", "mint", "pudina", "basil", "parsley", "dill", "shallot", "shallots",
        "sambar onion", "yam", "colocasia", "arbi", "taro", "sweet potato", "tapioca", "karela",
        "lauki", "tori", "brinjal", "eggplant", "baingan", "tindora", "parwal", "knol khol",
        "moringa", "sprouts", "kale", "bok choy", "arugula", "aloo", "pyaz", "tamatar", "gobi", "gajar"
    },
    "bakery": {
        "bread", "bun", "buns", "pav", "croissant", "croissants", "muffin", "muffins", "donut",
        "doughnut", "cake", "cakes", "pastry", "pastries", "pizza base", "pita", "naan", "roti",
        "chapati", "paratha", "baguette", "brioche", "sourdough", "toast", "garlic bread", "brown bread",
        "white bread", "multigrain", "burger bun", "dinner roll", "puff", "puffs", "tart", "pie",
        "brownie", "cupcake", "sponge cake", "plum cake", "focaccia", "ciabatta", "bagel"
    },
    "snacks": {
        "plantain", "banana chips", "tapioca chips", "rusk", "papad", "pappadam", "samosa", "vada",
        "bajji", "pakoda", "cutlet", "roll", "puff", "sandwich", "parotta", "dosa batter", "idli batter"
    }
}

# ================================================================
# PERISHABLE CATEGORIES (General Department Names)
# ================================================================
PERISHABLE_CATEGORIES = {
    "dairy", "milk", "curd", "cheese", "paneer", "butter", "ghee", "yogurt", "yoghurt",
    "meat", "fish", "seafood", "poultry", "chicken", "mutton", "pork", "beef", "lamb", "prawn", "shrimp", "crab",
    "fruit", "fruits", "fresh fruit",
    "vegetable", "vegetables", "veg", "greens", "leafy", "produce", "fresh produce",
    "bakery", "bread", "baked", "confectionery", "cake", "pastry",
    "egg", "eggs",
    "fresh", "perishable", "chilled", "frozen", "refrigerated", "cold chain", "deli",
    "snacks fresh", "plantain", "banana",
    "juice", "fresh juice", "smoothie",
    "prepared food", "ready to eat", "rte", "cooked",
    "salad", "cut fruit", "cut vegetable",
    "flowers", "floral",
}

# ================================================================
# NON-PERISHABLE CATEGORIES
# ================================================================
NON_PERISHABLE_CATEGORIES = {
    "grocery", "provisions", "staples", "dry goods", "kirana",
    "rice", "wheat", "flour", "atta", "maida", "sooji", "rava", "semolina",
    "dal", "pulses", "lentil", "lentils", "toor", "moong", "chana", "urad", "masoor",
    "oil", "cooking oil", "edible oil", "mustard oil", "coconut oil", "sunflower oil", "refined oil",
    "spice", "spices", "masala", "whole spice", "ground spice", "turmeric", "chilli powder", "coriander powder",
    "sugar", "jaggery", "sweetener",
    "salt", "rock salt", "table salt",
    "tea", "coffee", "beverage", "beverages",
    "biscuit", "biscuits", "cookies", "packaged snacks", "chips", "namkeen", "mixture", "murukku",
    "noodles", "pasta", "instant noodles", "maggi",
    "sauce", "ketchup", "jam", "pickle", "pickles", "achar",
    "canned", "canned food", "tinned",
    "cleaning", "detergent", "soap", "washing", "dish wash", "floor cleaner", "phenyl", "bleach", "disinfectant",
    "toiletries", "toilet", "personal care", "hygiene", "sanitary", "toothpaste", "toothbrush", "shampoo", "conditioner",
    "cosmetics", "beauty", "makeup", "skincare", "cream", "lotion", "sunscreen",
    "household", "home care", "utensils", "kitchen tools", "plastic", "aluminium foil", "cling wrap",
    "stationery", "office", "pen", "pencil", "notebook", "paper",
    "hardware", "electrical", "batteries", "bulb", "led",
    "baby care", "diaper", "diapers", "wipes",
    "pet food", "dog food", "cat food",
    "clothing", "apparel", "footwear", "textile",
    "medicine", "pharmaceutical", "pharma", "otc",
    "puja", "worship", "agarbatti", "incense", "camphor",
    "dry fruit", "dry fruits", "nuts", "seeds", "almond", "cashew", "walnut", "raisin",
    "water", "mineral water", "packaged water", "soda", "soft drink", "cold drink", "cola",
    "tobacco", "cigarette", "gutka", "beedi",
}

# ================================================================
# PERISHABLE PRODUCT NAME KEYWORDS
# Exhaustive dictionary of perishable groceries across all departments.
# ================================================================
PERISHABLE_PRODUCT_KEYWORDS = {
    # Dairy
    "milk", "curd", "yogurt", "yoghurt", "paneer", "cheese", "butter unsalted", "cream", "khoa", "khoya",
    "lassi", "buttermilk", "chaas", "thayir", "thyr", "dahi", "mozzarella", "ricotta", "cottage cheese",
    "milma", "amul", "nandini", "mother dairy", "verka", "milk mist", "sour cream",
    # Meat & Fish & Seafood
    "chicken", "mutton", "beef", "pork", "lamb", "fish", "prawn", "prawns", "shrimp", "shrimps", "crab", "crabs", "lobster",
    "squid", "octopus", "clam", "mussel", "oyster", "salmon", "tuna", "pomfret", "rohu", "catla", "seer", "kingfish",
    "mackerel", "sardine", "anchovy", "tilapia", "basa", "carp", "hilsa", "trout", "cod",
    "cutlet", "kebab", "kabab", "keema", "kheema", "mince", "sausage", "salami", "bacon", "ham", "pepperoni",
    "liver", "kidney", "wings", "breast", "drumstick", "thigh", "steak", "fillet", "duck", "turkey", "quail",
    "broiler", "meat curry", "fish curry", "boneless",
    # Fruits
    "mango", "apple", "orange", "banana", "grape", "grapes", "pineapple", "papaya", "guava", "pomegranate",
    "watermelon", "melon", "cantaloupe", "kiwi", "strawberry", "strawberries", "blueberry", "blueberries",
    "raspberry", "cherry", "cherries", "plum", "peach", "apricot", "lychee", "chiku", "chikoo", "sapota",
    "custard apple", "sitaphal", "jackfruit", "fig", "dragonfruit", "passion fruit", "starfruit", "pear", "avocado",
    "lemon", "lime", "mosambi", "sweet lime", "grapefruit", "gooseberry", "amla",
    "poka", "nendran", "robusta", "njali", "tender coconut",
    # Vegetables
    "tomato", "potato", "onion", "garlic", "ginger", "carrot", "beetroot", "radish", "turnip",
    "cabbage", "cauliflower", "broccoli", "spinach", "palak", "methi", "fenugreek", "lettuce",
    "capsicum", "bell pepper", "chilly", "chilli", "pepper", "cucumber", "zucchini", "pumpkin",
    "bottle gourd", "bitter gourd", "ridge gourd", "snake gourd", "ash gourd", "ivy gourd",
    "drumstick", "ladyfinger", "okra", "bhindi", "beans", "french beans", "cluster beans",
    "peas", "green peas", "corn", "sweet corn", "baby corn", "mushroom", "mushrooms",
    "spring onion", "leek", "celery", "asparagus", "artichoke",
    "coriander", "mallilla", "curry leaves", "mint", "pudina", "basil", "parsley", "dill",
    "ulli", "shallot", "shallots", "big onion", "small onion",
    "yam", "colocasia", "arbi", "taro", "sweet potato", "tapioca",
    "karela", "lauki", "tori", "brinjal", "eggplant", "baingan", "tindora", "sprouts",
    # Bakery & Fresh Baked
    "bread", "bun", "buns", "pav", "croissant", "muffin", "donut", "doughnut", "cake", "pastry", "pastries",
    "pizza base", "naan", "roti", "chapati", "paratha", "pappdam", "papad", "baguette", "brownie",
    # Eggs
    "egg", "eggs", "quail egg", "duck egg",
    # Fresh Prepared & Deli
    "cutlet", "samosa", "vada", "bajji", "pakoda", "biryani", "curry", "gravy",
    "parotta", "dosa batter", "idli batter", "fresh juice", "smoothie",
    "salad", "coleslaw", "sandwich", "wrap", "roll",
    # Fresh Snacks
    "rusk", "banana chips", "tapioca chips",
    # Tofu & Plant Based
    "tofu", "tempeh", "soy milk",
}

# ================================================================
# NON-PERISHABLE PRODUCT NAME KEYWORDS
# Strictly rejected regardless of what dropdown category was picked.
# ================================================================
NON_PERISHABLE_PRODUCT_KEYWORDS = {
    # Sanitizers, Disinfectants & Antiseptics
    "sanitizer", "sanitiser", "hand sanitizer", "hand sanitiser", "handwash", "hand wash",
    "disinfectant", "antiseptic", "savlon", "dettol", "rubbing alcohol", "sterillium",
    "sanitizer spray", "sanitizer gel", "surface cleaner", "lizol", "harpic", "domex",
    "bleach", "phenyl", "colin", "glass cleaner", "floor cleaner", "toilet cleaner",

    # Soaps & Bathing
    "soap", "lux", "dove", "pears", "lifebuoy", "hamam", "cinthol", "medimix", "mysore sandal",
    "fiama", "santoor", "liril", "margo", "body wash", "shower gel", "bath bomb",

    # Hair Care & Grooming
    "shampoo", "conditioner", "hair oil", "parachute", "dabur", "head shoulders", "pantene", "sunsilk",
    "tresemme", "clinic plus", "garnier", "hair color", "hair dye", "hair gel", "hair wax", "hair serum",

    # Oral Care
    "toothpaste", "toothbrush", "colgate", "pepsodent", "closeup", "sensodyne", "oral b", "mouthwash",
    "listerine", "dental floss", "tongue cleaner",

    # Skincare & Cosmetics
    "face wash", "facewash", "face cream", "moisturizer", "moisturiser", "lotion", "body lotion",
    "vaseline", "nivea", "pond", "ponds", "fair lovely", "glow lovely", "sunscreen", "cold cream", "lip balm",
    "boroline", "boroplus", "makeup", "lipstick", "foundation", "compact", "eyeliner", "mascara",
    "kajal", "nail polish", "nail paint", "remover", "concealer", "blush", "skincare", "cosmetic",

    # Shaving & Personal Hygiene
    "razor", "blade", "gillette", "shaving cream", "shaving foam", "aftershave",
    "sanitary pad", "stayfree", "whisper", "sofy", "tampon", "pantyliner",
    "deodorant", "perfume", "body spray", "axe", "fogg", "wild stone", "engage", "talcum", "powder",
    "comb", "hair band", "clip", "pin",

    # Cleaning & Laundry Detergents
    "detergent", "washing powder", "surf excel", "surf", "ariel", "tide", "rin", "nirma", "ghadi", "henko",
    "fabric conditioner", "comfort", "liquid detergent", "fabric wash",
    "dish wash", "dishwash", "dishwashing bar", "dishwashing liquid", "vim", "pril", "exo",
    "scrub", "scrubber", "scotch brite", "sponge", "steel wool", "mop", "broom", "dustpan",

    # Pest Control & Home Fragrance
    "mosquito coil", "hit", "allout", "mortein", "good knight", "baygon", "rat killer", "insect killer",
    "naphthalene", "naphthalene balls", "camphor tablets", "air freshener", "odonil", "godrej aer",

    # Wraps, Bags & Disposables
    "garbage bag", "dustbin", "trash bag", "aluminium foil", "aluminum foil", "cling wrap", "zipper bag", "ziploc",

    # Hardware, Electronics, Plastics & Furniture
    "battery", "batteries", "duracell", "eveready", "panasonic battery",
    "bulb", "led", "tube light", "torch", "candle", "matchbox", "lighter",
    "umbrella", "raincoat", "cloth", "towel", "bedsheet", "pillow", "blanket",
    "bottle", "water bottle", "flask", "thermos", "container", "plastic box", "tupperware",
    "peeler", "knife", "scissors", "fork", "spoon", "plate", "bowl", "cup", "mug", "glass",
    "pan", "pot", "cooker", "tava", "kadai", "utensil",
    "wire", "charger", "cable", "adapter", "plug", "socket", "extension cord", "tape", "glue", "fevicol", "fevikwik",
    "chair", "table", "desk", "furniture", "stool", "sofa", "fan", "heater", "iron", "mobile", "phone", "laptop", "tablet pc", "headphone", "earphone", "electronic", "electronics",

    # Stationery & Office
    "pen", "pencil", "eraser", "sharpener", "ruler", "scale", "notebook", "book", "diary", "register", "paper", "a4 paper", "envelope", "stapler", "staples", "marker", "highlighter",

    # Baby & Pet Care (Non-perishable)
    "diaper", "diapers", "pampers", "huggies", "mamy poko", "baby wipes", "wipes", "wet wipes", "cotton buds",
    "dog food", "cat food", "pedigree", "whiskas", "drools", "royal canin",

    # Medical & Pharma
    "medicine", "tablet", "tablets", "capsule", "capsules", "syrup", "paracetamol", "crocin", "dolo",
    "calpol", "combiflam", "aspirin", "disprin", "vicks", "inhaler", "balm", "iodex", "moov", "volini",
    "zandu balm", "tiger balm", "bandage", "bandaid", "cotton roll", "gauze", "antiseptic liquid",
    "cough syrup", "benadryl", "corex", "digene", "gelusil", "pan d", "pantocid", "antacid", "eno",
    "ors", "electral", "thermometer", "oximeter", "mask", "n95", "surgical mask",

    # Dry Provisions & Staples (Long Shelf Life)
    "rice", "basmati", "sona masoori", "ponni", "matta rice",
    "wheat", "atta", "maida", "sooji", "rava", "semolina", "besan", "gram flour",
    "dal", "toor dal", "moong dal", "chana dal", "urad dal", "masoor dal", "rajma", "chole", "chana", "kabuli chana",
    "poha", "flattened rice", "oats", "muesli", "cornflakes",
    "sugar", "jaggery", "honey", "sugar free", "sweetener",
    "salt", "rock salt", "iodised salt", "black salt",
    "sunflower oil", "groundnut oil", "mustard oil", "refined oil", "olive oil", "sesame oil", "gingelly oil",

    # Spices & Condiments (Shelf Stable)
    "turmeric", "chilli powder", "coriander powder", "cumin seed", "jeera", "pepper powder", "garam masala",
    "cinnamon", "cardamom", "clove", "nutmeg", "bay leaf", "fenugreek seed", "fennel seed", "star anise", "hing", "asafoetida",
    "everest", "mdh", "eastern", "sakthi masala",

    # Packaged Snacks & Biscuits
    "biscuit", "biscuits", "cookies", "parle", "britannia", "sunfeast", "oreo", "bourbon", "marie", "good day", "hide seek", "monaco", "krackjack",
    "chips", "lays", "kurkure", "bingo", "pringles", "doritos",
    "namkeen", "haldiram", "bikaji", "mixture", "sev", "murukku", "bhujia",

    # Shelf-stable Beverages
    "tea", "coffee", "green tea", "brook bond", "tata tea", "red label", "taj mahal", "nescafe", "bru", "davidoff",
    "horlicks", "bournvita", "complan", "boost", "milo",
    "coca cola", "pepsi", "sprite", "fanta", "thums up", "limca", "mirinda", "7up", "mountain dew", "red bull",
    "mineral water", "bisleri", "aquafina", "kinley", "packaged water", "tang", "rasna",

    # Instant Noodles & Pasta
    "maggi", "yippee", "top ramen", "wai wai", "knorr", "chings", "noodles", "pasta", "macaroni", "spaghetti", "vermicelli", "sevai",

    # Shelf-stable Sauces & Preserves
    "ketchup", "tomato sauce", "soy sauce", "chilli sauce", "vinegar", "mayonnaise",
    "jam", "marmalade", "peanut butter",
    "pickle", "pickles", "achar", "chutney powder",
    "canned", "tin", "sardine tin", "tuna tin",

    # Dry Fruits & Nuts
    "almond", "cashew", "walnut", "pistachio", "raisin", "dates", "dry dates", "kishmish", "kaju", "badam",

    # Puja & Miscellaneous
    "agarbatti", "incense", "camphor", "dhoop",
    "cigarette", "beedi", "gutka", "pan masala",
}


# Words representing baked fresh goods that may be made from flour/wheat
BAKERY_GRAIN_EXCEPTIONS = {
    "bread", "bun", "buns", "pav", "cake", "cakes", "pastry", "pastries",
    "roti", "paratha", "naan", "kulcha", "bagel", "croissant", "muffin", "toast", "puffs"
}


def _matches_word(keyword: str, text: str) -> bool:
    """
    Matches keyword as a complete word/phrase, allowing natural English plural suffixes (-s, -es, -ies).
    Prevents partial substring collisions (e.g. 'oil' in 'broiler', 'tea' in 'steam').
    """
    if keyword.endswith('y') and len(keyword) > 2 and keyword[-2] not in 'aeiou':
        stem = re.escape(keyword[:-1])
        pattern = r'(?:\b|_)(?:' + re.escape(keyword) + r'(?:s)?|' + stem + r'ies)(?:\b|_)'
    else:
        pattern = r'(?:\b|_)' + re.escape(keyword) + r'(?:s|es)?(?:\b|_)'
    return bool(re.search(pattern, text, re.IGNORECASE))


def classify_item(
    product_name: str,
    category: str = "",
    shelf_life_days: int = 0,
    has_expiry_date: bool = False
) -> Tuple[bool, str, float]:
    """
    Classifies a single product as perishable or non-perishable.
    Strictly prevents non-perishables (sanitizer, soap, electronics, toiletries)
    from being registered into the EcoPrice monitoring pipeline.

    Returns:
        (is_perishable: bool, reason: str, confidence: float 0.0-1.0)
    """
    name_clean = product_name.strip()
    name_lower = name_clean.lower()
    cat_clean = category.strip()
    cat_lower = cat_clean.lower() if cat_clean else ""

    if not name_clean:
        return False, "Product name cannot be empty", 1.0

    # ── Signal 1: Product Name Non-Perishable Override (Highest Priority) ──
    for keyword in sorted(NON_PERISHABLE_PRODUCT_KEYWORDS, key=len, reverse=True):
        if _matches_word(keyword, name_lower):
            # Exception: fresh baked items made from grains (e.g. 'Whole Wheat Bread', 'Atta Bun')
            if keyword in {"wheat", "atta", "maida", "sooji", "rava", "flour"} and any(_matches_word(bw, name_lower) for bw in BAKERY_GRAIN_EXCEPTIONS):
                continue
            return False, f"Product '{name_clean}' identified as non-perishable ('{keyword}')", 0.99

    # ── Signal 2: Category Department Non-Perishable Match ──
    if cat_lower:
        for keyword in sorted(NON_PERISHABLE_CATEGORIES, key=len, reverse=True):
            if _matches_word(keyword, cat_lower) or keyword in cat_lower:
                return False, f"Category '{category}' is a non-perishable department ('{keyword}')", 0.98

    # ── Signal 3: Product Name Known Perishable Match ──
    for keyword in sorted(PERISHABLE_PRODUCT_KEYWORDS, key=len, reverse=True):
        if _matches_word(keyword, name_lower):
            return True, f"Product '{name_clean}' matches perishable food item ('{keyword}')", 0.98

    # ── Signal 4: Category Plausibility Validation for Specific Perishable Departments ──
    # If the user selected a specific perishable category (Meat, Dairy, Fruits, Vegetables, etc.),
    # the product name MUST match that category's known vocabulary.
    target_lexicon = None
    if any(k in cat_lower for k in ["meat", "fish", "seafood", "poultry"]):
        target_lexicon = ("Meat & Fish", CATEGORY_LEXICONS["meat & fish"])
    elif any(k in cat_lower for k in ["dairy", "milk"]):
        target_lexicon = ("Dairy", CATEGORY_LEXICONS["dairy"])
    elif any(k in cat_lower for k in ["fruit"]):
        target_lexicon = ("Fruits", CATEGORY_LEXICONS["fruits"])
    elif any(k in cat_lower for k in ["veg", "produce", "greens"]):
        target_lexicon = ("Vegetables", CATEGORY_LEXICONS["vegetables"])
    elif any(k in cat_lower for k in ["baker", "bread"]):
        target_lexicon = ("Bakery", CATEGORY_LEXICONS["bakery"])
    elif any(k in cat_lower for k in ["snack", "plantain"]):
        target_lexicon = ("Snacks", CATEGORY_LEXICONS["snacks"])

    if target_lexicon:
        dept_name, lexicon = target_lexicon
        for kw in sorted(lexicon, key=len, reverse=True):
            if _matches_word(kw, name_lower):
                return True, f"Product '{name_clean}' verified as valid {dept_name} item ('{kw}')", 0.95
        # If specific department was chosen but product name does not match any recognized item in it
        return False, f"Product '{name_clean}' does not match recognized perishable items for category '{category}'", 0.95

    # ── Signal 5: General Fresh Departments (e.g. 'Specialty & Fresh') ──
    if cat_lower and any(k in cat_lower for k in ["fresh", "chilled", "refrigerated", "specialty"]):
        if 0 < shelf_life_days <= 14 or has_expiry_date:
            return True, f"Specialty fresh item with short shelf life ({shelf_life_days}d) admitted", 0.75
        elif shelf_life_days > 14:
            return False, f"Item shelf life ({shelf_life_days} days) exceeds perishable limit", 0.85

    # ── Signal 6: Shelf life heuristic fallback ──
    if shelf_life_days > 0:
        if shelf_life_days <= 7:
            return True, f"Short shelf life ({shelf_life_days} days) suggests perishable item", 0.70
        else:
            return False, f"Shelf life ({shelf_life_days} days) indicates non-perishable item", 0.85

    if has_expiry_date:
        return True, "Item has explicit near-term expiry date from supplier", 0.65

    # ── Final Fallback: Unrecognized items are rejected to prevent non-perishables in inventory ──
    return False, f"Product '{name_clean}' cannot be verified as a recognized perishable grocery item", 0.90


def classify_batch(items: list) -> dict:
    """
    Classifies an entire delivery manifest.

    Args:
        items: list of dicts with keys 'product', optionally 'category', 'shelf_life', 'has_expiry'

    Returns:
        {
            "perishable": [list of admitted items with classification metadata],
            "non_perishable": [list of rejected items with classification metadata],
            "summary": {
                "total": int,
                "perishable_count": int,
                "non_perishable_count": int
            }
        }
    """
    perishable = []
    non_perishable = []

    for item in items:
        is_perish, reason, confidence = classify_item(
            product_name=item.get("product", ""),
            category=item.get("category", ""),
            shelf_life_days=item.get("shelf_life", 0),
            has_expiry_date=item.get("has_expiry", False)
        )

        classified = {**item, "is_perishable": is_perish, "classification_reason": reason, "confidence": confidence}

        if is_perish:
            perishable.append(classified)
        else:
            non_perishable.append(classified)

    return {
        "perishable": perishable,
        "non_perishable": non_perishable,
        "summary": {
            "total": len(items),
            "perishable_count": len(perishable),
            "non_perishable_count": len(non_perishable),
        }
    }
