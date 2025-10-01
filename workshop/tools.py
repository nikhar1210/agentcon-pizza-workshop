def calculate_pizza_for_people(people_count: int, appetite_level: str = "normal") -> str:
    """
    Calculate the number and size of pizzas needed for a group of people.
    
    Args:
        people_count (int): Number of people who will be eating
        appetite_level (str): Appetite level - "light", "normal", or "heavy" (default: "normal")
    
    Returns:
        str: Recommendation for pizza size and quantity
    """
    if people_count <= 0:
        return "Please provide a valid number of people (greater than 0)."
    
    # Base calculations assuming normal appetite
    # Small pizza: 1-2 people
    # Medium pizza: 2-3 people  
    # Large pizza: 3-4 people
    # Extra Large pizza: 4-6 people
    
    appetite_multipliers = {
        "light": 0.7,
        "normal": 1.0,
        "heavy": 1.3
    }
    
    multiplier = appetite_multipliers.get(appetite_level.lower(), 1.0)
    adjusted_people = people_count * multiplier
    
    recommendations = []
    
    if adjusted_people <= 2:
        if adjusted_people <= 1:
            recommendations.append("1 Small pizza (perfect for 1-2 people)")
        else:
            recommendations.append("1 Medium pizza (great for 2-3 people)")
    elif adjusted_people <= 4:
        recommendations.append("1 Large pizza (serves 3-4 people)")
    elif adjusted_people <= 6:
        recommendations.append("1 Extra Large pizza (feeds 4-6 people)")
    elif adjusted_people <= 8:
        recommendations.append("2 Large pizzas (perfect for sharing)")
    elif adjusted_people <= 12:
        recommendations.append("2 Extra Large pizzas (great for groups)")
    else:
        # For larger groups, calculate multiple pizzas
        extra_large_count = int(adjusted_people // 5)
        remainder = adjusted_people % 5
        
        pizza_list = []
        if extra_large_count > 0:
            pizza_list.append(f"{extra_large_count} Extra Large pizza{'s' if extra_large_count > 1 else ''}")
        
        if remainder > 2:
            pizza_list.append("1 Large pizza")
        elif remainder > 0:
            pizza_list.append("1 Medium pizza")
        
        recommendations.append(" + ".join(pizza_list))
    
    result = f"For {people_count} people with {appetite_level} appetite:\n"
    result += f"🍕 Recommendation: {recommendations[0]}\n"
    
    if appetite_level != "normal":
        result += f"(Adjusted for {appetite_level} appetite level)"
    
    return result