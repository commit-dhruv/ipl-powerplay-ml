# src/team_names.py

TEAM_NAME_MAP = {
    "Delhi Daredevils":             "Delhi Capitals",
    "Kings XI Punjab":              "Punjab Kings",
    "Rising Pune Supergiant":       "Rising Pune Supergiants",
    "Royal Challengers Bangalore":  "Royal Challengers Bengaluru",
}

VENUE_NAME_MAP = {
    # Wankhede
    "Wankhede Stadium, Mumbai":                                        "Wankhede Stadium",
    # Chinnaswamy
    "M. Chinnaswamy Stadium":                                          "M Chinnaswamy Stadium",
    "M Chinnaswamy Stadium, Bengaluru":                                "M Chinnaswamy Stadium",
    # Mohali
    "Punjab Cricket Association IS Bindra Stadium":                    "Punjab Cricket Association Stadium, Mohali",
    "Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh":"Punjab Cricket Association Stadium, Mohali",
    "IS Bindra Stadium, Mohali":                                       "Punjab Cricket Association Stadium, Mohali",
    # Kotla / Arun Jaitley
    "Feroz Shah Kotla":                                                "Arun Jaitley Stadium",
    "Feroz Shah Kotla Ground":                                         "Arun Jaitley Stadium",
    "Arun Jaitley Stadium, Delhi":                                     "Arun Jaitley Stadium",
    # Eden Gardens
    "Eden Gardens, Kolkata":                                           "Eden Gardens",
    # Chepauk
    "MA Chidambaram Stadium":                                          "MA Chidambaram Stadium, Chepauk",
    "MA Chidambaram Stadium, Chepauk, Chennai":                        "MA Chidambaram Stadium, Chepauk",
    # Hyderabad
    "Rajiv Gandhi International Stadium":                              "Rajiv Gandhi International Stadium, Uppal",
    "Rajiv Gandhi International Stadium, Uppal, Hyderabad":            "Rajiv Gandhi International Stadium, Uppal",
    # Pune
    "Maharashtra Cricket Association Stadium":                         "Maharashtra Cricket Association Stadium, Pune",
    # Sawai Mansingh
    "Sawai Mansingh Stadium, Jaipur":                                  "Sawai Mansingh Stadium",
    # Himachal Pradesh
    "Himachal Pradesh Cricket Association Stadium, Dharamsala":        "Himachal Pradesh Cricket Association Stadium",
    # DY Patil
    "Dr DY Patil Sports Academy":                                      "Dr DY Patil Sports Academy, Mumbai",
    # Brabourne
    "Brabourne Stadium":                                               "Brabourne Stadium, Mumbai",
    # Narendra Modi / Motera
    "Sardar Patel Stadium, Motera":                                    "Narendra Modi Stadium, Ahmedabad",
    # Vizag
    "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam": "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium",
    
    "M.Chinnaswamy Stadium":                                       "M Chinnaswamy Stadium",
    "Punjab Cricket Association IS Bindra Stadium, Mohali":        "Punjab Cricket Association Stadium, Mohali",
}

def normalise_team(name: str) -> str:
    """Map legacy franchise names to their current canonical name."""
    return TEAM_NAME_MAP.get(name, name)

def normalise_venue(name: str) -> str:
    """Map venue name variants to a single canonical name."""
    if not isinstance(name, str):
        return name
    return VENUE_NAME_MAP.get(name, name)