
cuisine_t={
    'italian':r'fettucini|rigatoni|lasagna|spaghetti|penne|gnocchi|tortellini|pasta|carbonara|pizza|calzone|garlic bread|alfredo|mozzarella|caesar|cacio|fe[t]{1,2}u[c]{1,2}ine|ravioli|burrata|proscuito|chicken parm|alfredo',
    'vietnamese': r'\b(pho)\b|spring roll|vietnamese|\b(ba[nh]{2})\b mi|thit nuong|cha gio|\bcuon\b|summer roll',
    'korean': r'bibimbap|korean|kimchi',
    'indian': r'paneer|tikka|masala|indian|pakora|gobhi|samosa|naan|basmati|lassi|saag|biryani|makhni|vindaloo|tandoori|korma|butter chicken|dolma',
    'southern': r'fried chicken|gumbo|brisket|smoke|bbq|fried zucchini|bbq|coleslaw',
    'mediterranean': r'pita|tabouleh|fattoush|gyro|kebab|kabob|skewer|falafel|greek|kofta|shawarma|hummus|tzatziki',
    'breakfast': r'egg|breakfast|bagel|toast|bacon|omelette|hash brown|croissant|lox|waffle|pancake|sausage',
    'american': r'mac.*cheese|burger|\bwing[s]?\b|bacon|reuben|cheesesteak|tater tots|fries|buffalo|ranch|onion rings|grilled cheese|melt|nashville|slider|chili cheese|garlic knots|tender|nuggets|dog',
    'chinese': r'orange chicken|tofu|chinese|mein|dumplings|mongolian|potsticker|fried rice|general tsos|wontons|chow fun|szechuan|beef broccoli|kung pao|\b(beef broccoli)\b',
    'japanese': r'ramen|sushi|sashimi|nigiri|unagi|katsu|((?<!egg)(?<!spring)(?<!lobster)(?<!lamb)(?<!curry)(?<!cinnamon)\sroll)|gyoza|tempura|miso|edamame|udon|wasabi|karaage|teriyaki|\bsoba\b',
    'mexican': r'mexican|taco|burrito|guac|chorizo|al pastor|quesadilla|salsa|birria|horchata|carne asada|el verde|refried beans|tostada|nachos|churro|tortillas',
    'latin': r'\barepa\b|empanada|jerk|caribbean',
    'thai': r'panang|pad thai|pad see ew|\bthai\b|drunken noodle|((red|yellow|green)\scurry)|tom kha|massaman|satay',
    'sandwiches': r'sandwich|blt|turkey club|roast beef',
    'soup': r'(soup)',
    'coffee': r'latte|capuccino|coffee|cappucino|cold brew',
    'drinks': r'water|coke|sprite|ginger ale|lemonade|pepsi|juice|\b(tea)\b|gatorade',
    'hawaiian': r'hawaiian|poke|musubi',
    'healthy': r'salad|juice|healthy|fruit|acai|berry|vegan|vegetables|veggies|smoothie',
    'sweets': r'waffle|ice cream|tiramisu|oreo|cinnamon roll|cheesecake|smoothie|donuts|chocolate|cookie|caramel|pudding',
    'seafood': r'fish|lobster|crab|shrimp',
    'rice': r'rice bowl|white rice',
}

metrics = {
            'Requested Orders': 'requested_orders',
            'Accepted Orders': 'accepted_orders',
            'Completed Orders': 'completed_orders',
            'First Time Orders': 'first_time_orders',
            'First Time Orders w/ Promo': 'first_time_orders_promo',
            'First Time Orders w/o Promo': 'first_time_order_organic',
            'Returning Orders': 'returning_orders',
            'Returning Orders w/ Promo': 'returning_orders_promo',
            'Returning Orders w/o Promo': 'returning_order_organic',
            'Promo Orders': 'total_orders_promo',
            'Order Issues': 'order_issues',
            'Eater Spend': 'total_eater_spend',
            'Eater Discount': 'total_eater_discount',
            'Eater Revenue': 'total_eater_revenue',
            'Average Prep Time in Min': 'avg_prep_time_min',
            'Average Rating': 'avg_rating',
            'Promo Order Rate': 'promo_order_rate',
            'First Time Order Rate': 'first_time_order_rate',
            'Returning Order Rate': 'returning_order_rate',
            '% of First Time On Promo': 'pct_first_time_promo',
            '% of Returning On Promo': 'pct_returning_promo',
            'Acceptance Rate': 'acceptance_rate',
            'Completion Rate': 'completion_rate',
            'Promo ROI': 'promo_roi',
            #'Revenue Per Prep Second': 'revenue_per_prep_min',
            #'% Chg over Time Split': '%_chg',
           }




#metrics = dict(sum_metrics.items() + extra_metrics.items() + average_metrics.items())
food_types = {
    'Food Item Type': 'item_type_new',
    'Cuisine Type': 'tags'
}

calc_metrics = {
            'requested_orders': 'sum',
            'accepted_orders': 'sum',
            'completed_orders': 'sum',
            'first_time_orders': 'sum',
            'first_time_orders_promo': 'sum',
            'returning_orders': 'sum',
            'returning_orders_promo': 'sum',
            'total_orders_promo': 'sum',
            'order_issues': 'sum',
            'total_eater_spend': 'sum',
            'total_eater_discount': 'sum',
            'total_eater_revenue': 'sum',
            'avg_prep_time_min': 'mean',
            'avg_rating': 'mean',
}